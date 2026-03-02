import os
import jax
import jax.numpy as jnp
import optax
import sys
# Add the repository root directory to sys.path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pinncoat.geometry_utils import load_stl_surface, load_cathode_nas, generate_fluid_points
from src.pinncoat.network import PotentialPINN, init_network
from src.pinncoat.train import train_model

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

    print("Initializing network...")
    # Network Initialization
    key = jax.random.PRNGKey(42)
    variables = init_network(key, input_shape=(1, 3), L_char=L_char, V_scale=v_anode, center=center)

    # Extract params from initialized variables
    # init_network returns the full variables dict. The model uses variable collections,
    # typically "params"
    params = variables['params'] if 'params' in variables else variables

    model = PotentialPINN(L_char=L_char, V_scale=v_anode, center=center)

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
    }

    print("Starting training...")
    # Execution
    state, losses = train_model(
        model=model,
        params=params,
        tx=tx,
        batch_data=batch_data,
        epochs=5000,
        mode="fixed"
    )

    print("Training completed.")

if __name__ == "__main__":
    main()
