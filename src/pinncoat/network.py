import jax.numpy as jnp
import flax.linen as nn

class PotentialPINN(nn.Module):
    """
    Physics-Informed Neural Network for predicting electric potential.
    """
    x_min: jnp.ndarray
    x_max: jnp.ndarray
    V0: float = 100.0
    features: list[int] = (64, 64, 64, 64)
    L_fourier: int = 4

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (jnp.ndarray): Input array of shape (N, 3) representing spatial coordinates.

        Returns:
            jnp.ndarray: Output array of shape (N, 1) representing electric potential.
        """
        # Dynamic Min-Max Scaling: Maps physical [x_min, x_max] to exactly [-1, 1]
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0

        features_list = [x_norm]
        for i in range(self.L_fourier):
            freq = (2.0 ** i) * jnp.pi
            features_list.append(jnp.sin(freq * x_norm))
            features_list.append(jnp.cos(freq * x_norm))

        F = jnp.concatenate(features_list, axis=-1)

        U = nn.Dense(self.features[0])(F)
        U = nn.tanh(U)
        V = nn.Dense(self.features[0])(F)
        V = nn.tanh(V)

        H = nn.Dense(self.features[0])(F)
        H = nn.tanh(H)

        for feat in self.features[1:]:
            H = nn.Dense(feat)(H)
            H = nn.tanh(H)
            H = H * U + V

        # Final output layer without activation function
        z = nn.Dense(1)(H)

        # Physical Output Bounds: Sigmoid forces output between [0, 1]
        # Multiplying by V0 scales it to exactly [0, V0], preventing negative potentials
        phi_norm = nn.sigmoid(z)
        phi_physical = phi_norm * self.V0

        return phi_physical

def init_network(rng_key, x_min, x_max, V0=100.0, input_shape=(1, 3), L_fourier=4):
    """
    Initializes the PotentialPINN network.

    Args:
        rng_key (jax.random.PRNGKey): JAX random key for initialization.
        x_min (jnp.ndarray): Minimum spatial bounds.
        x_max (jnp.ndarray): Maximum spatial bounds.
        V0 (float): Maximum physical voltage.
        input_shape (tuple): Shape of the input data. Defaults to (1, 3).
        L_fourier (int): Number of frequency bands for Fourier Features.

    Returns:
        dict: Initialized network variables (weights).
    """
    model = PotentialPINN(x_min=x_min, x_max=x_max, V0=V0, L_fourier=L_fourier)
    dummy_input = jnp.zeros(input_shape)
    variables = model.init(rng_key, dummy_input)
    return variables
