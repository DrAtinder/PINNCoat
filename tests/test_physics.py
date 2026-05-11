import unittest
import jax
import jax.numpy as jnp
from src.pinncoat.network import PotentialPINN, init_network
from src.pinncoat.physics import energy_loss, dirichlet_loss, robin_loss, compute_total_loss

class TestPhysicsLosses(unittest.TestCase):

    def setUp(self):
        self.rng_key = jax.random.PRNGKey(0)
        self.model = PotentialPINN()
        self.params = init_network(self.rng_key, input_shape=(1, 3))

        # Dummy data
        self.N = 10
        key1, key2, key3, key4 = jax.random.split(self.rng_key, 4)
        self.x_fluid = jax.random.normal(key1, (self.N, 3))
        self.x_anode = jax.random.normal(key2, (self.N, 3))
        self.x_cathode = jax.random.normal(key3, (self.N, 3))
        self.normals = jax.random.normal(key4, (self.N, 3))
        # Normalize normals
        self.normals = self.normals / jnp.linalg.norm(self.normals, axis=1, keepdims=True)

        self.v_anode = 100.0
        self.v_cathode = 0.0
        self.r_film = 5.0
        self.sigma = 0.1

    def test_energy_loss_shape(self):
        loss = energy_loss(self.params, self.model, self.x_fluid)
        self.assertEqual(loss.shape, ())
        self.assertTrue(jnp.isfinite(loss))

    def test_dirichlet_loss_shape(self):
        loss = dirichlet_loss(self.params, self.model, self.x_anode, self.v_anode)
        self.assertEqual(loss.shape, ())
        self.assertTrue(jnp.isfinite(loss))

    def test_robin_loss_shape(self):
        loss = robin_loss(self.params, self.model, self.x_cathode, self.normals,
                          self.v_cathode, self.r_film, self.sigma)
        self.assertEqual(loss.shape, ())
        self.assertTrue(jnp.isfinite(loss))

    def test_compute_total_loss_shape(self):
        loss, aux = compute_total_loss(self.params, self.model, self.x_fluid, self.x_anode,
                                  self.x_cathode, self.normals, self.v_anode, self.v_cathode,
                                  self.r_film, self.sigma)
        self.assertEqual(loss.shape, ())
        self.assertTrue(jnp.isfinite(loss))
        self.assertEqual(len(aux), 5)

if __name__ == '__main__':
    unittest.main()
