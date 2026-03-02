import unittest
import jax.random
import jax.numpy as jnp
from src.pinncoat.network import PotentialPINN, init_network

class TestPotentialPINN(unittest.TestCase):
    def test_network_output_shape(self):
        # Initialize random key
        rng_key = jax.random.PRNGKey(0)

        # Initialize the network
        variables = init_network(rng_key, input_shape=(10, 3))

        # Create dummy point cloud array of shape (10, 3)
        dummy_points = jax.random.normal(rng_key, (10, 3))

        # Initialize the model
        model = PotentialPINN()

        # Pass dummy points through the network
        output = model.apply(variables, dummy_points)

        # Assert the output shape is exactly (10, 1)
        self.assertEqual(output.shape, (10, 1))

if __name__ == "__main__":
    unittest.main()
