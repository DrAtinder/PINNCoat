import jax.numpy as jnp
import flax.linen as nn

class PotentialPINN(nn.Module):
    """
    Physics-Informed Neural Network for predicting electric potential.
    """
    features: list[int] = (64, 64, 64, 64)

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (jnp.ndarray): Input array of shape (N, 3) representing spatial coordinates.

        Returns:
            jnp.ndarray: Output array of shape (N, 1) representing electric potential.
        """
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)

        # Final output layer without activation function
        x = nn.Dense(1)(x)
        return x

def init_network(rng_key, input_shape=(1, 3)):
    """
    Initializes the PotentialPINN network.

    Args:
        rng_key (jax.random.PRNGKey): JAX random key for initialization.
        input_shape (tuple): Shape of the input data. Defaults to (1, 3).

    Returns:
        dict: Initialized network variables (weights).
    """
    model = PotentialPINN()
    dummy_input = jnp.zeros(input_shape)
    variables = model.init(rng_key, dummy_input)
    return variables
