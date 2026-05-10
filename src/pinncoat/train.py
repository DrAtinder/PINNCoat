import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import struct
from src.pinncoat.physics import compute_total_loss, energy_loss, dirichlet_loss, robin_loss


@jax.jit
def train_step_fixed(state, batch_data, weights):
    """
    Standard training step with fixed penalty weights.
    """
    def loss_fn(params):
        return compute_total_loss({'params': params}, state.apply_fn, **batch_data, weights=weights)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (loss_e, loss_d, loss_r, loss_s)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, loss_e, loss_d, loss_r, loss_s


@jax.jit
def train_step_adaptive(state, batch_data, dynamic_weights):
    """
    Training step with adaptive loss balancing.
    """
    def losses_fn(params):
        loss_e = energy_loss({'params': params}, state.apply_fn, batch_data['x_fluid'])
        loss_d = dirichlet_loss({'params': params}, state.apply_fn, batch_data['x_anode'], batch_data['v_anode'])
        loss_r = robin_loss({'params': params}, state.apply_fn, batch_data['x_cathode'], batch_data['normals'],
                            batch_data['v_cathode'], batch_data['r_film'], batch_data['sigma'])
        return loss_e, loss_d, loss_r

    # Compute individual losses
    loss_e, loss_d, loss_r = losses_fn(state.params)

    # Simple heuristic to balance weights: inverse running average of loss magnitudes (or just current magnitudes)
    # Using 1 / (loss + 1e-8)
    new_weights = 1.0 / (jnp.array([loss_e, loss_d, loss_r]) + 1e-8)
    # Normalize weights so they sum to 3.0 (or similar) to keep overall learning rate scale
    new_weights = 3.0 * new_weights / jnp.sum(new_weights)

    # Smooth update of dynamic_weights using EMA (Exponential Moving Average)
    alpha = 0.9
    updated_weights = alpha * dynamic_weights + (1 - alpha) * new_weights

    def total_loss_fn(params):
        return compute_total_loss({'params': params}, state.apply_fn, **batch_data, weights=updated_weights)

    grad_fn = jax.value_and_grad(total_loss_fn, has_aux=True)
    (total_loss, (loss_e, loss_d, loss_r, loss_s)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, total_loss, updated_weights, loss_e, loss_d, loss_r, loss_s


class TrainStateLagrange(train_state.TrainState):
    """
    Custom TrainState for Lagrange multipliers.
    """
    lambda_d: jnp.ndarray
    lambda_r: jnp.ndarray
    opt_state_lambdas: optax.OptState
    tx_lambdas: optax.GradientTransformation = struct.field(pytree_node=False)

@jax.jit
def train_step_lagrange(state, batch_data):
    """
    Min-max step for Lagrange multipliers.
    """
    def total_loss_fn(params, lambda_d, lambda_r):
        loss_e = energy_loss({'params': params}, state.apply_fn, batch_data['x_fluid'])
        loss_d = dirichlet_loss({'params': params}, state.apply_fn, batch_data['x_anode'], batch_data['v_anode'])
        loss_r = robin_loss({'params': params}, state.apply_fn, batch_data['x_cathode'], batch_data['normals'],
                            batch_data['v_cathode'], batch_data['r_film'], batch_data['sigma'])
        # Return total loss, and individual boundary losses for lambda gradients
        return loss_e + lambda_d * loss_d + lambda_r * loss_r, (loss_d, loss_r)

    # 1. Gradient Descent step: minimize total loss w.r.t network params
    def loss_for_params(params):
        loss, _ = total_loss_fn(params, state.lambda_d, state.lambda_r)
        return loss

    loss, grads = jax.value_and_grad(loss_for_params)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # 2. Gradient Ascent step: maximize total loss w.r.t lambdas (which is just the boundary losses)
    # The gradient of L w.r.t lambda_d is exactly loss_d, w.r.t lambda_r is loss_r.
    # To maximize, we need to take a step in the direction of the positive gradient.
    # We negate the gradient because Optax assumes minimization.
    _, (loss_d, loss_r) = total_loss_fn(state.params, state.lambda_d, state.lambda_r)

    grad_lambdas = {
        'lambda_d': -loss_d,
        'lambda_r': -loss_r
    }

    lambdas = {
        'lambda_d': state.lambda_d,
        'lambda_r': state.lambda_r
    }

    lambda_updates, new_opt_state_lambdas = state.tx_lambdas.update(
        grad_lambdas, state.opt_state_lambdas, lambdas
    )

    new_lambdas = optax.apply_updates(lambdas, lambda_updates)

    # Ensure lambdas remain positive
    new_lambda_d = jnp.maximum(0.0, new_lambdas['lambda_d'])
    new_lambda_r = jnp.maximum(0.0, new_lambdas['lambda_r'])

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        lambda_d=new_lambda_d,
        lambda_r=new_lambda_r,
        opt_state_lambdas=new_opt_state_lambdas
    )

    return new_state, loss


def train_model(model, params, tx, batch_data, epochs=100, mode="fixed", weights=(1.0, 1.0, 1.0, 100.0), tx_lambdas=None):
    """
    Main training wrapper for PINNCoat.

    Args:
        model: Flax neural network model.
        params: Initial network parameters.
        tx: Optax optimizer for network parameters.
        batch_data: Dictionary containing training data.
        epochs: Number of training epochs.
        mode: Training mode ('fixed', 'adaptive', 'lagrange').
        weights: Initial weights for losses (used in 'fixed' and 'adaptive' modes).
        tx_lambdas: Optax optimizer for Lagrange multipliers (required for 'lagrange' mode).

    Returns:
        Trained state and list of losses.
    """
    if mode == "lagrange":
        if tx_lambdas is None:
            tx_lambdas = optax.sgd(learning_rate=0.01) # Default for lagrange

        lambdas = {
            'lambda_d': jnp.array(weights[1]),
            'lambda_r': jnp.array(weights[2])
        }
        opt_state_lambdas = tx_lambdas.init(lambdas)

        state = TrainStateLagrange.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            lambda_d=lambdas['lambda_d'],
            lambda_r=lambdas['lambda_r'],
            opt_state_lambdas=opt_state_lambdas,
            tx_lambdas=tx_lambdas
        )
    else:
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )
        dynamic_weights = jnp.array(weights)
        fixed_weights = jnp.array(weights)

    losses = []

    for epoch in range(epochs):
        if mode == "fixed":
            state, loss, loss_e, loss_d, loss_r, loss_s = train_step_fixed(state, batch_data, fixed_weights)
        elif mode == "adaptive":
            state, loss, dynamic_weights, loss_e, loss_d, loss_r, loss_s = train_step_adaptive(state, batch_data, dynamic_weights)
        elif mode == "lagrange":
            # For brevity, lagrange does not return the individual components here unless modified as well.
            # Assuming 'fixed' or 'adaptive' is mainly used based on the PR comment.
            state, loss = train_step_lagrange(state, batch_data)
            loss_e = loss_d = loss_r = loss_s = 0.0 # Placeholder
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        losses.append(loss)

        if epoch % max(1, epochs // 10) == 0:
            if mode in ["fixed", "adaptive"]:
                print(f"Epoch {epoch}/{epochs}, Total: {loss:.2f} | Energy: {loss_e:.2f} | Anode: {loss_d:.2f} | Cathode: {loss_r:.2f} | Shield: {loss_s:.2f}")
            else:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")

            if mode == "adaptive":
                if len(dynamic_weights) > 3:
                    print(f"  Weights: E={dynamic_weights[0]:.2f}, D={dynamic_weights[1]:.2f}, R={dynamic_weights[2]:.2f}, S={dynamic_weights[3]:.2f}")
                else:
                    print(f"  Weights: {dynamic_weights}")
            elif mode == "lagrange":
                print(f"  Lambda D: {state.lambda_d:.4f}, Lambda R: {state.lambda_r:.4f}")

    return state, losses
