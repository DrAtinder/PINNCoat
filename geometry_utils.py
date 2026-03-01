import trimesh
import numpy as np
import meshio

def load_stl_surface(stl_path, num_points):
    """
    Load an STL surface, sample uniform points on the surface,
    and return the mesh, sampled points, and normals.

    Args:
        stl_path (str): Path to the STL file.
        num_points (int): Number of points to sample.

    Returns:
        tuple: (trimesh.Trimesh, np.ndarray, np.ndarray) representing
               the mesh object, the sampled points, and the normals at those points.
    """
    mesh = trimesh.load(stl_path)
    # trimesh.sample.sample_surface returns (points, face_indices)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # Extract the normals of the sampled faces
    normals = mesh.face_normals[face_indices]

    return mesh, points, normals


def load_cathode_nas(nas_path, num_points):
    """
    Read a Nastran file, extract surface shell elements (triangles/quads),
    create a Trimesh object, sample points, and calculate normals.

    Args:
        nas_path (str): Path to the Nastran (.nas) file.
        num_points (int): Number of points to sample.

    Returns:
        tuple: (trimesh.Trimesh, np.ndarray, np.ndarray) representing
               the mesh object, the sampled points, and the normals at those points.
    """
    mesh_data = meshio.read(nas_path)

    # We want to extract only the triangles and quads to make the surface mesh
    nodes = mesh_data.points
    faces = []

    for cell_block in mesh_data.cells:
        if cell_block.type == "triangle":
            faces.append(cell_block.data)
        elif cell_block.type == "quad":
            # Triangulate the quads: a quad (0, 1, 2, 3) becomes two triangles (0, 1, 2) and (0, 2, 3)
            quads = cell_block.data
            tri1 = quads[:, [0, 1, 2]]
            tri2 = quads[:, [0, 2, 3]]
            faces.append(tri1)
            faces.append(tri2)

    if not faces:
        raise ValueError("No surface elements (triangles or quads) found in the Nastran file.")

    faces = np.vstack(faces)

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=nodes, faces=faces)

    # Sample points on the surface
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_indices]

    return mesh, points, normals


def generate_fluid_points(bath_mesh, obstacle_meshes, num_points):
    """
    Sample random 3D points inside the bath bounding box.
    Filter these points so they are STRICTLY inside the bath mesh
    and STRICTLY OUTSIDE the obstacle meshes.

    Args:
        bath_mesh (trimesh.Trimesh): The bounding bath mesh.
        obstacle_meshes (list[trimesh.Trimesh]): List of obstacle meshes inside the bath.
        num_points (int): Target number of fluid points to generate.

    Returns:
        np.ndarray: Array of shape (num_points, 3) representing valid fluid points.
    """
    bounds = bath_mesh.bounds
    min_bound, max_bound = bounds[0], bounds[1]

    valid_points = []

    # To optimize sampling, we'll try fetching batches of points
    batch_size = num_points * 2

    while len(valid_points) < num_points:
        # Sample random points within the bounding box
        random_points = np.random.uniform(min_bound, max_bound, size=(batch_size, 3))

        # Check if points are inside the bath mesh
        # ray.contains returns True if a point is inside the volume
        inside_bath_mask = bath_mesh.ray.contains_points(random_points)

        # Keep only points inside the bath
        candidates = random_points[inside_bath_mask]

        # Check against each obstacle
        if len(candidates) > 0:
            outside_obstacles_mask = np.ones(len(candidates), dtype=bool)
            for obs_mesh in obstacle_meshes:
                # contains_points returns True if point is inside the obstacle
                inside_obs = obs_mesh.ray.contains_points(candidates)
                outside_obstacles_mask &= ~inside_obs

            accepted = candidates[outside_obstacles_mask]
            valid_points.extend(accepted)

    # Return exactly num_points
    return np.array(valid_points[:num_points])
