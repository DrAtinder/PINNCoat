import jax.numpy as jnp
import flax.linen as nn

class PotentialPINN(nn.Module):
    """
    Physics-Informed Neural Network for predicting electric potential.
    """
    features: list[int] = (64, 64, 64, 64)
    L_char: float = 1.0
    V_scale: float = 1.0
    center: tuple = (0.0, 0.0, 0.0)

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (jnp.ndarray): Input array of shape (N, 3) representing spatial coordinates.

        Returns:
            jnp.ndarray: Output array of shape (N, 1) representing electric potential.
        """
        c = jnp.array(self.center, dtype=x.dtype)
        x = (x - c) / self.L_char

        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)

        # Final output layer without activation function
        x = nn.Dense(1)(x)
        x = x * self.V_scale
        return x

def init_network(rng_key, input_shape=(1, 3), L_char=1.0, V_scale=1.0, center=(0.0, 0.0, 0.0)):
    """
    Initializes the PotentialPINN network.

    Args:
        rng_key (jax.random.PRNGKey): JAX random key for initialization.
        input_shape (tuple): Shape of the input data. Defaults to (1, 3).
        L_char (float): Characteristic length for spatial scaling.
        V_scale (float): Voltage scaling factor.
        center (tuple): Center of the spatial domain.

    Returns:
        dict: Initialized network variables (weights).
    """
    model = PotentialPINN(L_char=L_char, V_scale=V_scale, center=center)
    dummy_input = jnp.zeros(input_shape)
    variables = model.init(rng_key, dummy_input)
    return variables
