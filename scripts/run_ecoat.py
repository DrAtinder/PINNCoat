import os
import jax
import jax.numpy as jnp
import optax
import sys
import plotly.graph_objects as go
import pyvista as pv
import numpy as np
# Add the repository root directory to sys.path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pinncoat.geometry_utils import load_stl_surface, load_cathode_nas, generate_fluid_points
from src.pinncoat.network import PotentialPINN, init_network
from src.pinncoat.train import train_model

def export_for_paraview(model, params, fluid_points, cathode_points, out_dir="data/output"):
    os.makedirs(out_dir, exist_ok=True)

    # Predict potentials
    phi_fluid = model.apply({'params': params}, fluid_points)
    phi_cathode = model.apply({'params': params}, cathode_points)

    # Convert points to standard NumPy arrays
    f_points_np = np.asarray(fluid_points)
    c_points_np = np.asarray(cathode_points)

    # Convert potentials to 1D standard NumPy arrays
    f_phi_np = np.asarray(phi_fluid).flatten()
    c_phi_np = np.asarray(phi_cathode).flatten()

    # Create PyVista point clouds
    fluid_vtk = pv.PolyData(f_points_np)
    cathode_vtk = pv.PolyData(c_points_np)

    # Assign the potential data to the VTK objects
    fluid_vtk["Electric_Potential_V"] = f_phi_np
    cathode_vtk["Electric_Potential_V"] = c_phi_np

    # Save the files
    fluid_vtk.save(os.path.join(out_dir, "fluid_results.vtp"))
    cathode_vtk.save(os.path.join(out_dir, "cathode_results.vtp"))


def plot_results(model, params, fluid_points, cathode_points, L_char, center, V_scale):
    print("Predicting potentials for visualization...")
    phi_fluid = model.apply({'params': params}, fluid_points)
    phi_cathode = model.apply({'params': params}, cathode_points)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=fluid_points[:, 0],
        y=fluid_points[:, 1],
        z=fluid_points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=phi_fluid.flatten(),
            colorscale='Viridis',
            opacity=0.3,
            showscale=True,
            colorbar=dict(title="Fluid Pot", x=0.8)
        ),
        name="Fluid Potential"
    ))

    fig.add_trace(go.Scatter3d(
        x=cathode_points[:, 0],
        y=cathode_points[:, 1],
        z=cathode_points[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=phi_cathode.flatten(),
            colorscale='Inferno',
            showscale=True,
            colorbar=dict(title="Cathode Pot", x=0.9)
        ),
        name="Cathode Surface Potential"
    ))

    fig.update_layout(
        title="PINN predicted potential",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    fig.show()

def main():
    # Paths
    bath_path = "data/raw/bath.stl"
    anode_path = "data/raw/anode.stl"
    cathode_path = "data/raw/cathode.nas"

    # Physical Constants
    v_anode = 250.0
    v_cathode = 0.0
    sigma = 1.5
    r_film = 1.0

    print("Loading geometries...")
    # Geometry
    bath_mesh, bath_points, bath_normals = load_stl_surface(bath_path, 2000)

    bounds = bath_mesh.bounds
    center = tuple((bounds[0] + bounds[1]) / 2.0)
    L_char = float(jnp.max(bounds[1] - bounds[0]) / 2.0)

    anode_mesh, anode_points, anode_normals = load_stl_surface(anode_path, 2000)
    cathode_mesh, cathode_points, cathode_normals = load_cathode_nas(cathode_path, 4000)

    print("Generating fluid points...")
    fluid_points = generate_fluid_points(
        bath_mesh,
        obstacle_meshes=[anode_mesh, cathode_mesh],
        num_points=10000,
        cathode_points=cathode_points,
        cathode_normals=cathode_normals
    )

    print("Generating shield points...")
    shield_offset = 5.0
    shield_points = cathode_points - shield_offset * cathode_normals

    print("Initializing network...")
    # Network Initialization
    key = jax.random.PRNGKey(42)
    variables = init_network(key, input_shape=(1, 3), L_char=L_char, V_scale=v_anode, center=center, L_fourier=4)

    # Extract params from initialized variables
    # init_network returns the full variables dict. The model uses variable collections,
    # typically "params"
    params = variables['params'] if 'params' in variables else variables

    model = PotentialPINN(L_char=L_char, V_scale=v_anode, center=center, L_fourier=4)

    # Optimizer Setup
    tx = optax.adam(learning_rate=1e-3)

    print("Packaging data...")
    # Data Packaging
    batch_data = {
        'x_fluid': jnp.array(fluid_points, dtype=jnp.float32),
        'x_anode': jnp.array(anode_points, dtype=jnp.float32),
        'v_anode': jnp.array(v_anode, dtype=jnp.float32),
        'x_cathode': jnp.array(cathode_points, dtype=jnp.float32),
        'normals': jnp.array(cathode_normals, dtype=jnp.float32),
        'v_cathode': jnp.array(v_cathode, dtype=jnp.float32),
        'r_film': jnp.array(r_film, dtype=jnp.float32),
        'sigma': jnp.array(sigma, dtype=jnp.float32),
        'x_shield': jnp.array(shield_points, dtype=jnp.float32),
    }

    print("Starting training...")
    # Execution
    state, losses = train_model(
        model=model,
        params=params,
        tx=tx,
        batch_data=batch_data,
        epochs=5000,
        mode="fixed",
        weights=(1.0, 1.0, 1.0, 100.0)
    )

    print("Training completed.")

    export_for_paraview(model, state.params, fluid_points, cathode_points)

    plot_results(model, state.params, fluid_points, cathode_points, L_char, center, v_anode)

if __name__ == "__main__":
    main()
