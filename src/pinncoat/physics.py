import jax
import jax.numpy as jnp

def get_phi_and_grad(params, model, x):
    """
    Computes the predicted potential and its spatial gradient.
    """
    def phi_scalar(p, x_single):
        # Add a batch dimension to x_single, run model, and return scalar
        # model can be an object with .apply or a callable (like state.apply_fn)
        apply_fn = model.apply if hasattr(model, 'apply') else model
        return apply_fn(p, jnp.expand_dims(x_single, 0))[0, 0]

    # Calculate both value and gradient, then vmap over the batch dimension
    return jax.vmap(jax.value_and_grad(phi_scalar, argnums=1), in_axes=(None, 0))(params, x)

def laplace_loss(params, model, x_fluid):
    """Computes the Strong Form PDE residual: mean((nabla^2 phi)^2)"""
    def get_laplacian(x):
        def phi_single(xi):
            apply_fn = model.apply if hasattr(model, 'apply') else model
            return jnp.squeeze(apply_fn(params, xi.reshape(1, -1)))
        # jax.hessian computes the exact 3x3 matrix of second derivatives
        H = jax.hessian(phi_single)(x)
        return jnp.trace(H) # Trace is d2/dx2 + d2/dy2 + d2/dz2

    laplacians = jax.vmap(get_laplacian)(x_fluid)
    return jnp.mean(laplacians**2)

def energy_loss(params, model, x_fluid):
    """
    Computes the Deep Ritz energy functional over fluid points.
    """
    phi, grad_phi = get_phi_and_grad(params, model, x_fluid)
    return 0.5 * jnp.mean(jnp.sum(grad_phi**2, axis=1))

def dirichlet_loss(params, model, x_anode, v_anode):
    """
    Computes the Mean Squared Error for the Dirichlet boundary condition (Anode).
    """
    apply_fn = model.apply if hasattr(model, 'apply') else model
    phi_anode = apply_fn(params, x_anode)[:, 0]
    return jnp.mean((phi_anode - v_anode)**2)

def robin_loss(params, model, x_cathode, normals, v_cathode, r_film, sigma):
    """
    Computes the Mean Squared Error for the Robin boundary condition (Cathode).
    """
    phi, grad_phi = get_phi_and_grad(params, model, x_cathode)

    # phi has shape (N,) if phi_scalar returned a scalar. Wait, let's make it match (N, 1) if needed,
    # or just use broadcast.
    # The prompt: Compute the normal derivative: dphi_dn = jnp.sum(grad_phi * normals, axis=1, keepdims=True)
    dphi_dn = jnp.sum(grad_phi * normals, axis=1, keepdims=True)

    # Compute predicted current density
    J_pred = sigma * dphi_dn

    # Expand phi to match (N, 1)
    phi = phi.reshape(-1, 1)

    # Target current density
    J_target = (phi - v_cathode) / r_film

    return jnp.mean((J_pred - J_target)**2)

def shield_loss(params, model, x_shield, v_cathode):
    """
    Penalizes voltage inside the solid geometry to force the field to route through openings.
    """
    apply_fn = model.apply if hasattr(model, 'apply') else model
    phi_shield = apply_fn(params, x_shield)[:, 0]
    return jnp.mean((phi_shield - v_cathode)**2)

def sensor_loss(params, model, x_sensor, v_sensor_true):
    """
    Supervised MSE loss for sensor data assimilation.
    """
    apply_fn = model.apply if hasattr(model, 'apply') else model
    v_pred = apply_fn(params, x_sensor)[:, 0]
    return jnp.mean((v_pred - v_sensor_true)**2)

def compute_total_loss(params, model, x_fluid, x_anode, x_cathode, normals, v_anode, v_cathode, r_film, sigma, x_shield=None, x_sensor=None, v_sensor=None, weights=(1.0, 1.0, 1.0, 100.0), fluid_method="energy"):
    """
    Computes the total loss as a weighted sum of energy, Dirichlet, Robin, shield, and sensor losses.
    """
    if fluid_method == "laplace":
        loss_e = laplace_loss(params, model, x_fluid)
    else:
        loss_e = energy_loss(params, model, x_fluid)
    loss_d = dirichlet_loss(params, model, x_anode, v_anode)
    loss_r = robin_loss(params, model, x_cathode, normals, v_cathode, r_film, sigma)

    loss_s = 0.0
    weight_s = 0.0
    if x_shield is not None and len(weights) > 3:
        loss_s = shield_loss(params, model, x_shield, v_cathode)
        weight_s = weights[3]

    loss_sens = 0.0
    weight_sens = 0.0
    if x_sensor is not None and v_sensor is not None and len(weights) > 4:
        loss_sens = sensor_loss(params, model, x_sensor, v_sensor)
        weight_sens = weights[4]

    total_loss = weights[0] * loss_e + weights[1] * loss_d + weights[2] * loss_r + weight_s * loss_s + weight_sens * loss_sens
    return total_loss, (loss_e, loss_d, loss_r, loss_s, loss_sens)
