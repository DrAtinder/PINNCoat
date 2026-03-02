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


def generate_fluid_points(bath_mesh, obstacle_meshes, num_points, cathode_points=None, cathode_normals=None, boundary_layer_ratio=0.5):
    """
    Sample random 3D points inside the bath bounding box.
    Filter these points so they are STRICTLY inside the bath mesh
    and STRICTLY OUTSIDE the obstacle meshes.

    Args:
        bath_mesh (trimesh.Trimesh): The bounding bath mesh.
        obstacle_meshes (list[trimesh.Trimesh]): List of obstacle meshes inside the bath.
        num_points (int): Target number of fluid points to generate.
        cathode_points (np.ndarray, optional): Points on the cathode surface for boundary layer sampling.
        cathode_normals (np.ndarray, optional): Normals corresponding to the cathode points for outward extrusion.
        boundary_layer_ratio (float, optional): Ratio of points to sample in the boundary layer. Default is 0.5.

    Returns:
        np.ndarray: Array of shape (num_points, 3) representing valid fluid points.
    """
    bounds = bath_mesh.bounds
    min_bound, max_bound = bounds[0], bounds[1]

    if cathode_points is not None:
        num_boundary_points = int(num_points * boundary_layer_ratio)
        num_uniform_points = num_points - num_boundary_points
    else:
        num_boundary_points = 0
        num_uniform_points = num_points

    # Helper function to filter points
    def filter_points(points):
        inside_bath_mask = bath_mesh.ray.contains_points(points)
        candidates = points[inside_bath_mask]

        if len(candidates) > 0:
            outside_obstacles_mask = np.ones(len(candidates), dtype=bool)
            for obs_mesh in obstacle_meshes:
                inside_obs = obs_mesh.ray.contains_points(candidates)
                outside_obstacles_mask &= ~inside_obs
            return candidates[outside_obstacles_mask]
        return np.array([])

    valid_uniform_points = []
    batch_size_uniform = max(num_uniform_points * 2, 1)

    while len(valid_uniform_points) < num_uniform_points:
        # Sample random points within the bounding box
        random_points = np.random.uniform(min_bound, max_bound, size=(batch_size_uniform, 3))
        accepted = filter_points(random_points)
        valid_uniform_points.extend(accepted)

    valid_uniform_points = np.array(valid_uniform_points[:num_uniform_points])

    valid_boundary_points = []
    if num_boundary_points > 0 and cathode_points is not None and cathode_normals is not None and len(cathode_points) > 0:
        batch_size_boundary = max(num_boundary_points * 2, 1)
        while len(valid_boundary_points) < num_boundary_points:
            # Sample from cathode points and extrude outward along the normal
            indices = np.random.choice(len(cathode_points), size=batch_size_boundary)
            sampled_cathode = cathode_points[indices]
            sampled_normals = cathode_normals[indices]

            push_distance = np.abs(np.random.normal(scale=10.0, size=(batch_size_boundary, 1)))
            extruded_points = sampled_cathode + (sampled_normals * push_distance)

            accepted = filter_points(extruded_points)
            valid_boundary_points.extend(accepted)

        valid_boundary_points = np.array(valid_boundary_points[:num_boundary_points])

    if len(valid_boundary_points) > 0:
        if len(valid_uniform_points) > 0:
            final_points = np.concatenate((valid_uniform_points, valid_boundary_points), axis=0)
        else:
            final_points = valid_boundary_points
    else:
        final_points = valid_uniform_points

    # Shuffle the concatenated array
    np.random.shuffle(final_points)

    return final_points
