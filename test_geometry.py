import unittest
import os
import numpy as np
import trimesh
from geometry_utils import load_stl_surface, load_cathode_nas, generate_fluid_points

class TestGeometryUtils(unittest.TestCase):
    def setUp(self):
        self.bath_path = "data/raw/bath.stl"
        self.anode_path = "data/raw/anode.stl"
        self.cathode_path = "data/raw/cathode.nas"

    def test_load_stl_surface(self):
        num_points = 100
        mesh, points, normals = load_stl_surface(self.bath_path, num_points)

        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertEqual(points.shape, (num_points, 3))
        self.assertEqual(normals.shape, (num_points, 3))

        # Normals should be roughly normalized
        norms = np.linalg.norm(normals, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-3))

    def test_load_cathode_nas(self):
        num_points = 50
        mesh, points, normals = load_cathode_nas(self.cathode_path, num_points)

        self.assertIsInstance(mesh, trimesh.Trimesh)
        self.assertEqual(points.shape, (num_points, 3))
        self.assertEqual(normals.shape, (num_points, 3))

        # Ensure it has faces
        self.assertTrue(len(mesh.faces) > 0)

    def test_generate_fluid_points(self):
        bath_mesh, _, _ = load_stl_surface(self.bath_path, 10)
        anode_mesh, _, _ = load_stl_surface(self.anode_path, 10)
        cathode_mesh, _, _ = load_cathode_nas(self.cathode_path, 10)

        obstacle_meshes = [anode_mesh, cathode_mesh]
        num_fluid_points = 50

        fluid_points = generate_fluid_points(bath_mesh, obstacle_meshes, num_fluid_points)

        self.assertEqual(fluid_points.shape, (num_fluid_points, 3))

        # Verify points are strictly inside bath
        inside_bath = bath_mesh.ray.contains_points(fluid_points)
        self.assertTrue(np.all(inside_bath))

        # Verify points are strictly outside obstacles
        for obs_mesh in obstacle_meshes:
            inside_obs = obs_mesh.ray.contains_points(fluid_points)
            self.assertFalse(np.any(inside_obs))

if __name__ == '__main__':
    unittest.main()
