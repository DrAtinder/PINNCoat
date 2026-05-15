import jax
import jax.numpy as jnp
import optax
import jaxopt
from flax.training import train_state
from flax import struct
from functools import partial
from src.pinncoat.physics import compute_total_loss, energy_loss, dirichlet_loss, robin_loss, laplace_loss


@partial(jax.jit, static_argnames=['fluid_method'])
def train_step_fixed(state, batch_data, weights, fluid_method="energy"):
    """
    Standard training step with fixed penalty weights.
    """
    def loss_fn(params):
        return compute_total_loss({'params': params}, state.apply_fn, **batch_data, weights=weights, fluid_method=fluid_method)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (loss_e, loss_d, loss_r, loss_s, loss_sens)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, loss_e, loss_d, loss_r, loss_s, loss_sens


@partial(jax.jit, static_argnames=['fluid_method'])
def train_step_adaptive(state, batch_data, dynamic_weights, fluid_method="energy"):
    """
    Training step with adaptive loss balancing.
    """
    def losses_fn(params):
        if fluid_method == "laplace":
            loss_e = laplace_loss({'params': params}, state.apply_fn, batch_data['x_fluid'])
        else:
            loss_e = energy_loss({'params': params}, state.apply_fn, batch_data['x_fluid'])
        loss_d = dirichlet_loss({'params': params}, state.apply_fn, batch_data['x_anode'], batch_data['v_anode'])
        loss_r = robin_loss({'params': params}, state.apply_fn, batch_data['x_cathode'], batch_data['normals'],
                            batch_data['v_cathode'], batch_data['r_film'], batch_data['sigma'])
        return loss_e, loss_d, loss_r

    # Compute individual losses
    loss_e, loss_d, loss_r = losses_fn(state.params)

    # Calculate dynamic weights based on the number of provided dynamic weights
    num_weights = dynamic_weights.shape[0]

    # We balance the core 3 losses if we only have 3 or 4 weights (shield is usually fixed)
    # If we have 5 weights (sensor added), we just use the first 3 for balancing here as a simple heuristic,
    # and keep the others fixed, or we could balance them all. But the current losses_fn only returns 3.
    # To avoid changing the logic too much, we will update the first 3 and leave the rest as is.
    core_losses = jnp.array([loss_e, loss_d, loss_r])
    new_core_weights = 1.0 / (core_losses + 1e-8)
    new_core_weights = 3.0 * new_core_weights / jnp.sum(new_core_weights)

    # Pad new_weights to match the shape of dynamic_weights
    if num_weights > 3:
        new_weights = jnp.concatenate([new_core_weights, dynamic_weights[3:]])
    else:
        new_weights = new_core_weights

    # Smooth update of dynamic_weights using EMA (Exponential Moving Average)
    alpha = 0.9
    updated_weights = alpha * dynamic_weights + (1 - alpha) * new_weights

    def total_loss_fn(params):
        return compute_total_loss({'params': params}, state.apply_fn, **batch_data, weights=updated_weights, fluid_method=fluid_method)

    grad_fn = jax.value_and_grad(total_loss_fn, has_aux=True)
    (total_loss, (loss_e, loss_d, loss_r, loss_s, loss_sens)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, total_loss, updated_weights, loss_e, loss_d, loss_r, loss_s, loss_sens


class TrainStateLagrange(train_state.TrainState):
    """
    Custom TrainState for Lagrange multipliers.
    """
    lambda_d: jnp.ndarray
    lambda_r: jnp.ndarray
    opt_state_lambdas: optax.OptState
    tx_lambdas: optax.GradientTransformation = struct.field(pytree_node=False)

@partial(jax.jit, static_argnames=['fluid_method'])
def train_step_lagrange(state, batch_data, fluid_method="energy"):
    """
    Min-max step for Lagrange multipliers.
    """
    def total_loss_fn(params, lambda_d, lambda_r):
        # We manually construct the loss to handle lagrange tracking properly.
        # It's better to just reuse compute_total_loss to maintain consistency.
        # But lagrange uses lambda_d and lambda_r as weights for Dirichlet and Robin.
        # We also need to support shield loss if it's there.
        # Let's set weights to use the lagrange multipliers, and 1.0 for energy, and 100.0 for shield if present.
        weights = (1.0, lambda_d, lambda_r, 100.0, 500.0) # Assuming shield and sensor weights are fixed
        total_loss, (loss_e, loss_d, loss_r, loss_s, loss_sens) = compute_total_loss({'params': params}, state.apply_fn, **batch_data, weights=weights, fluid_method=fluid_method)
        return total_loss, (loss_e, loss_d, loss_r, loss_s, loss_sens)

    # 1. Gradient Descent step: minimize total loss w.r.t network params
    def loss_for_params(params):
        return total_loss_fn(params, state.lambda_d, state.lambda_r)

    grad_fn = jax.value_and_grad(loss_for_params, has_aux=True)
    (loss, (loss_e, loss_d, loss_r, loss_s, loss_sens)), grads = grad_fn(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # 2. Gradient Ascent step: maximize total loss w.r.t lambdas (which is just the boundary losses)
    # The gradient of L w.r.t lambda_d is exactly loss_d, w.r.t lambda_r is loss_r.
    # To maximize, we need to take a step in the direction of the positive gradient.
    # We negate the gradient because Optax assumes minimization.

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

    return new_state, loss, loss_e, loss_d, loss_r, loss_s, loss_sens


def train_model(model, params, tx, batch_data, epochs=100, mode="fixed", weights=(1.0, 1.0, 1.0, 100.0), tx_lambdas=None, fluid_method="energy"):
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
        fluid_method: string method for fluid loss calculation ('energy' or 'laplace').

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
            state, loss, loss_e, loss_d, loss_r, loss_s, loss_sens = train_step_fixed(state, batch_data, fixed_weights, fluid_method=fluid_method)
        elif mode == "adaptive":
            state, loss, dynamic_weights, loss_e, loss_d, loss_r, loss_s, loss_sens = train_step_adaptive(state, batch_data, dynamic_weights, fluid_method=fluid_method)
        elif mode == "lagrange":
            state, loss, loss_e, loss_d, loss_r, loss_s, loss_sens = train_step_lagrange(state, batch_data, fluid_method=fluid_method)
        else:
            raise ValueError(f"Unknown training mode: {mode}")

        losses.append(loss)

        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}/{epochs}, Total: {loss:.2f} | Energy: {loss_e:.2f} | Anode: {loss_d:.2f} | Cathode: {loss_r:.2f} | Shield: {loss_s:.2f} | Sensor: {loss_sens:.2f}")

            if mode == "adaptive":
                if len(dynamic_weights) > 3:
                    print(f"  Weights: E={dynamic_weights[0]:.2f}, D={dynamic_weights[1]:.2f}, R={dynamic_weights[2]:.2f}, S={dynamic_weights[3]:.2f}")
                else:
                    print(f"  Weights: {dynamic_weights}")
            elif mode == "lagrange":
                print(f"  Lambda D: {state.lambda_d:.4f}, Lambda R: {state.lambda_r:.4f}")

    return state, losses

def train_lbfgs(model, init_params, batch_data, weights, fluid_method, maxiter=5000):
    """
    Phase 2: L-BFGS Optimization to resolve stiff boundaries.
    """
    print("Starting Phase 2: L-BFGS Optimization...")

    # Define a wrapper that returns ONLY the total loss (a scalar) for jaxopt
    def objective_fn(params):
        # Call the existing compute_total_loss function
        loss, _ = compute_total_loss({'params': params}, model.apply, **batch_data, weights=weights, fluid_method=fluid_method)
        return loss

    # Initialize the L-BFGS optimizer via ScipyMinimize
    lbfgs = jaxopt.ScipyMinimize(fun=objective_fn, method="L-BFGS-B", maxiter=maxiter)

    # Run the optimizer starting from the final Adam weights
    state = lbfgs.run(init_params)

    print(f"L-BFGS finished with status: {state.state.status}")
    return state.params
