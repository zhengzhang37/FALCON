import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
from flax import nnx
from typing import Sequence, NamedTuple, Any, Dict, Tuple
import random


def generate_fatigue_params(fatigue_params: Dict[str, float], sequence_length: int, key: jax.random.PRNGKey) -> Dict[str, jax.Array]:
    """
    Generates fatigue vectors for the human model.
    Rules for fatigue vectors:
    - p_init: Initial performance level, should be between 0 and 1.
    - p_max: Maximum performance level, should be between 0 and 1.
    - p_min: Minimum performance level, should be between 0 and 1.
    - epsilon_peak: Workload at which performance peaks.
    - epsilon_mid: Workload at which performance is halfway between p_max and p_min during decline.
    - p_decay: Decay rate of performance, should be a positive value.
    All values should be numpy arrays for compatibility with JAX operations.
    
    Ranges and Steps:
    0.7 <= p_init <= 0.9 (step 0.05)
    0.4 <= p_min <= 0.5 (step 0.05)
    0.8 <= p_max <= 1.0 (step 0.05)
    p_init <= p_max
    5 <= epsilon_peak <= 20 (step 5)
    50 <= epsilon_mid <= 100 (step 5)
    0.05 <= p_decay <= 0.1 (step 0.01)
    """

    keys = jax.random.split(key, 6)
    # Generate p_max first to ensure the p_init <= p_max constraint can be met.
    p_max = jax.random.choice(keys[0], jnp.arange(fatigue_params['p_max_min'], fatigue_params['p_max_max'], 0.05))

    # Generate p_init ensuring it is less than or equal to p_max.
    p_init_options = jnp.arange(fatigue_params['p_init_min'], fatigue_params['p_init_max'], 0.05)
    p_init = jax.random.choice(keys[1], p_init_options)
    p_init = jnp.minimum(p_init, p_max)

    p_min = jax.random.choice(keys[2], jnp.arange(fatigue_params['p_min_min'], fatigue_params['p_min_max'], 0.05))
    epsilon_peak = jax.random.choice(keys[3], jnp.arange(fatigue_params['epsilon_peak_min'] * sequence_length, fatigue_params['epsilon_peak_max'] * sequence_length, 5, dtype=jnp.float32))
    epsilon_mid = jax.random.choice(keys[4], jnp.arange(fatigue_params['epsilon_mid_min'] * sequence_length, fatigue_params['epsilon_mid_max'] * sequence_length, 5, dtype=jnp.float32))
    p_decay = jax.random.choice(keys[5], jnp.arange(fatigue_params['p_decay_min'], fatigue_params['p_decay_max'], 0.01))

    return {
        'p_init': p_init,
        'p_max': p_max,
        'p_min': p_min,
        'epsilon_peak': epsilon_peak,
        'epsilon_mid': epsilon_mid,
        'p_decay': p_decay
    }
class Human(nnx.Module):
    """Models the human expert with fatigue-dependent classification noise."""
    def __init__(self, num_classes: int, fatigue_params: Dict[str, float], sequence_length: int, *, rngs: nnx.Rngs):
        self.num_classes = num_classes
        self.fatigue_params = generate_fatigue_params(fatigue_params, sequence_length, rngs.params())
        self.rngs = rngs

    def fatigue_function(
        self, 
        epsilon: jax.Array, 
        ) -> jax.Array:
        """Calculates fatigue based on the provided parameters."""
        f_vecs = self.fatigue_params
        
        def true_fun():
            return f_vecs['p_init'] + (f_vecs['p_max'] - f_vecs['p_init']) * (epsilon / f_vecs['epsilon_peak']) ** 2

        def false_fun():
            return f_vecs['p_min'] + (f_vecs['p_max'] - f_vecs['p_min']) / (1 + jnp.exp(f_vecs['p_decay'] * (epsilon - f_vecs['epsilon_mid'])))

        true_result = true_fun()
        false_result = false_fun()

        return jnp.where(
            epsilon < f_vecs['epsilon_peak'],
            true_result,
            false_result,
        )

    def get_noise_rate(self, fatigue_counter: jax.Array) -> jax.Array:
        
        fatigue_level = self.fatigue_function(fatigue_counter)
        return 1.0 - fatigue_level

    def __call__(self, true_label: jax.Array, fatigue_counter: jax.Array, *, rngs: nnx.Rngs) -> jax.Array:
        """Produces a potentially noisy classification based on the true label and fatigue."""
        key = rngs.dropout()
        eta = self.get_noise_rate(fatigue_counter.squeeze(1))
        prob_correct = 1.0 - eta
        y_onehot = jax.nn.one_hot(true_label, self.num_classes)
        prob_vector = y_onehot * prob_correct[:, None] + (1.0 - y_onehot) * (eta[:, None] / (self.num_classes - 1))
        predicted_label = jax.random.categorical(key, logits=jnp.log(prob_vector))
        predicted_label = jax.nn.one_hot(predicted_label, self.num_classes)
        return predicted_label


