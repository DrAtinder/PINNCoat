# PINNCoat ⚡🚗

**A Mesh-Free Electrodeposition Solver using Physics-Informed Neural Networks in JAX.**

PINNCoat is a high-performance, purely mesh-free solver designed to simulate electrodeposition (Ecoat) processes, potential propagation, and dynamic film thickness growth. Instead of relying on traditional finite volume or finite element meshes, PINNCoat utilizes the **Deep Ritz Method** (Variational PINNs) powered by JAX to solve the elliptic Laplace equation directly on point clouds sampled from STL geometries.

## 🚀 Key Features
* **Strictly Mesh-Free:** Abandons complex volumetric meshing. Domain and boundaries are represented entirely by spatial coordinates $(x, y, z)$ sampled directly from CAD (STL) files.
* **Deep Ritz Formulation:** Solves the Laplace equation by minimizing the variational energy functional rather than the PDE residual, requiring only first-order derivatives (`jax.grad`) for massive speedups and training stability.
* **Natural Adiabatic Boundaries:** Zero-flux boundaries (the bath walls) are handled naturally by the energy functional, completely removing them from the loss calculation.
* **Dynamic Thickness Growth:** Film growth is modeled as a time-varying Robin boundary condition at the cathode, seamlessly handling moving boundaries without topological remeshing.
* **JAX-Accelerated:** Fully JIT-compiled for extreme performance on CPU, GPU, or TPU architectures.

## 🧠 Methodology

### The Forward Problem
The electric potential $\phi$ is approximated by a Multi-Layer Perceptron (MLP) built with Flax. The network minimizes the energy functional (weak form) over the fluid domain:

$$\mathcal{L}_{Energy} = \int_{\Omega} \frac{1}{2} \sigma |\nabla \phi|^2 d\Omega$$

Coupled with a penalty for the Dirichlet boundary (Anode) and a dynamic Robin boundary condition (Cathode) incorporating film resistance $R_{film}$.

### Inverse & Parameter Optimization
Because the entire solver is differentiable end-to-end, arbitrary electrical resistance models (non-linear film resistance, overpotential) can be embedded directly into the loss function and optimized against experimental thickness data using automatic differentiation.

## 🗺️ Project Roadmap
- [ ] **Task 1:** Geometry processing and uniform point sampling from Bath, Anode, and Cathode STLs using `trimesh`.
- [ ] **Task 2:** Define the JAX/Flax MLP architecture.
- [ ] **Task 3:** Implement the Deep Ritz energy functional and boundary loss functions.
- [ ] **Task 4:** Construct the highly optimized XLA training loop using Optax.
- [ ] **Task 5:** Implement the outer time-stepping loop for current density calculation and thickness growth.

## 🛠️ Requirements
* `jax`
* `flax`
* `optax`
* `trimesh`
* `numpy`
