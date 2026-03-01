import plotly.graph_objects as go
from geometry_utils import load_stl_surface, load_cathode_nas, generate_fluid_points

def main():
    bath_path = "data/raw/bath.stl"
    anode_path = "data/raw/anode.stl"
    cathode_path = "data/raw/cathode.nas"

    # Load and sample 1000 points from each boundary
    print("Loading boundaries...")
    bath_mesh, bath_points, bath_normals = load_stl_surface(bath_path, 1000)
    anode_mesh, anode_points, anode_normals = load_stl_surface(anode_path, 1000)
    cathode_mesh, cathode_points, cathode_normals = load_cathode_nas(cathode_path, 1000)

    # Generate 3000 fluid points
    print("Generating fluid points...")
    obstacle_meshes = [anode_mesh, cathode_mesh]
    fluid_points = generate_fluid_points(bath_mesh, obstacle_meshes, 3000)

    # Create Plotly traces
    print("Creating visualization...")

    # Bath trace: light blue, low opacity
    bath_trace = go.Scatter3d(
        x=bath_points[:, 0], y=bath_points[:, 1], z=bath_points[:, 2],
        mode='markers',
        name='Bath',
        marker=dict(size=3, color='lightblue', opacity=0.3)
    )

    # Anode trace: red
    anode_trace = go.Scatter3d(
        x=anode_points[:, 0], y=anode_points[:, 1], z=anode_points[:, 2],
        mode='markers',
        name='Anode',
        marker=dict(size=4, color='red', opacity=0.8)
    )

    # Cathode trace: black
    cathode_trace = go.Scatter3d(
        x=cathode_points[:, 0], y=cathode_points[:, 1], z=cathode_points[:, 2],
        mode='markers',
        name='Cathode',
        marker=dict(size=4, color='black', opacity=0.8)
    )

    # Fluid trace: green, small markers
    fluid_trace = go.Scatter3d(
        x=fluid_points[:, 0], y=fluid_points[:, 1], z=fluid_points[:, 2],
        mode='markers',
        name='Fluid',
        marker=dict(size=2, color='green', opacity=0.5)
    )

    fig = go.Figure(data=[bath_trace, anode_trace, cathode_trace, fluid_trace])

    # Ensure 1:1:1 aspect ratio
    fig.update_layout(
        scene=dict(aspectmode='data'),
        title="PINNCoat Geometry Verification",
        margin=dict(l=0, r=0, b=0, t=30)
    )

    output_file = "geometry_visualization.html"
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    main()
