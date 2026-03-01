# PINNCoat System Architecture & Instructions

You are acting as the lead AI co-developer for PINNCoat, a mesh-free electrodeposition (Ecoat) solver. 

## Core Tech Stack
* **Language:** Python 3.10+
* **ML Framework:** JAX, Flax, Optax
* **Geometry Processing:** Trimesh, NumPy

## Architectural Rules (STRICT)
1. **Mesh-Free Paradigm:** Do not use finite volume, finite element, or Cartesian grid discretizations for the fluid domain. The domain is represented exclusively by unorganized point clouds $(x, y, z)$.
2. **Deep Ritz Method:** We are solving the Laplace equation using the Variational PINN (Deep Ritz) formulation. **DO NOT use `jax.hessian`** to compute second derivatives for the PDE residual. Instead, use `jax.grad` to compute first-order derivatives and minimize the energy functional: 0.5 * sigma * |grad(phi)|^2.
3. **Boundary Conditions:** * Bath walls are adiabatic (natural boundary condition; do not enforce explicitly).
   * Anodes use Dirichlet boundary conditions (enforced via MSE penalty).
   * Cathodes use Robin boundary conditions (to handle dynamic film resistance and thickness growth).
4. **Performance:** Ensure all forward passes, loss calculations, and training steps are pure functions and compiled using `@jax.jit`.
5. **Code Style:** Write modular, highly documented, and type-hinted Python code. Keep geometry processing separate from the neural network definitions and the training loop.
