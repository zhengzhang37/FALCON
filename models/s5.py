from functools import partial
import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import lecun_normal, normal
from jax import random
from jax.numpy.linalg import eigh
import numpy as np
class SequenceLayer(nnx.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            prenorm     (bool):     apply prenorm if true or postnorm if false
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    def __init__(
        self,
        ssm: nnx.Module,
        d_model: int,
        activation: str = 'gelu',
        do_norm: bool = True,
        prenorm: bool = True,
        do_gtrxl_norm: bool = True,
        step_rescale: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.ssm = ssm
        self.d_model = d_model
        self.activation = activation
        self.prenorm = prenorm
        self.do_norm = do_norm
        self.do_gtrxl_norm = do_gtrxl_norm
        self.step_rescale = step_rescale 

        # Initialize ssm
        self.seq = ssm(step_rescale=step_rescale, rngs=rngs)
        if self.activation in ["full_glu"]:
            self.out1 = nnx.Linear(self.d_model, self.d_model, rngs=rngs)
            self.out2 = nnx.Linear(self.d_model, self.d_model, rngs=rngs)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nnx.Linear(self.d_model, self.d_model, rngs=rngs)

        self.norm = nnx.LayerNorm(self.d_model, rngs=rngs)

    def __call__(self, hidden, x, d):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             hidden (complex64): Hidden state carry, shape (1, batch_size, state_size).
             x (float32):      Input sequence, shape (L, d_model).
             d (bool):         Reset signal, shape (L,).
        Returns:
            A tuple containing the new hidden state and the output sequence (L, d_model).
        """
        skip = x
        if self.prenorm and self.do_norm:
            x = self.norm(x)
        hidden, x = jax.vmap(self.seq, in_axes=1, out_axes=1)(hidden, x, d)
        if self.do_gtrxl_norm:
            x = self.norm(x)

        if self.activation == "full_glu":
            x = nnx.gelu(x)
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
        elif self.activation == "half_glu1":
            x = nnx.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x))
        elif self.activation == "half_glu2":
            x1 = nnx.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x1))
        elif self.activation == "gelu":
            x = nnx.gelu(x)
        else:
            raise NotImplementedError(f"Activation: {self.activation} not implemented")

        x = skip + x
        if not self.prenorm and self.do_norm:
            x = self.norm(x)

        return hidden, x

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return jnp.zeros((1, batch_size, hidden_size), dtype=jnp.float16)

def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(key, shape):
        """ Init function
             Args:
                 key: jax random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    """ Initialize an array of learnable timescale parameters
         Args:
             key: jax random key
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H,)
     """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)


def init_VinvB(init_fun, rng, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: jax random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return np.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
    """
    Identity = jnp.ones(Lambda.shape[0], dtype=Lambda.dtype)

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
    """
    Identity = jnp.ones(Lambda.shape[0], dtype=Lambda.dtype)
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

# Parallel scan operations
@jax.vmap
def binary_operator_reset(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i, c_i = q_i
    A_j, b_j, c_j = q_j
    return (
        (A_j * A_i)*(1 - c_j) + A_j * c_j,
        (A_j * b_i + b_j)*(1 - c_j) + b_j * c_j,
        c_i * (1 - c_j) + c_j,
    )



def apply_ssm(Lambda_bar, B_bar, C_tilde, hidden, input_sequence, resets, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            reset      (bool): input sequence of features                (L,)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)
    Lambda_elements = jnp.concatenate([
        jnp.ones((1, Lambda_bar.shape[0])),
        Lambda_elements,
    ])

    Bu_elements = jnp.concatenate([
        hidden,
        Bu_elements,
    ])

    if resets is None:
        _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    else:
        resets = jnp.concatenate([
            jnp.zeros(1),
            resets,
        ])
        _, xs, _ = jax.lax.associative_scan(binary_operator_reset, (Lambda_elements, Bu_elements, resets))
    xs = xs[1:]

    if conj_sym:
        return xs[np.newaxis, -1], jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return xs[np.newaxis, -1], jax.vmap(lambda x: (C_tilde @ x).real)(xs)

class S5SSM(nnx.Module):
    def __init__(
        self,
        Lambda_re_init: jnp.ndarray,
        Lambda_im_init: jnp.ndarray,
        V: jnp.ndarray,
        Vinv: jnp.ndarray,
        H: int,
        P: int,
        C_init: str,
        discretization: str,
        dt_min: float,
        dt_max: float,
        conj_sym: bool = True,
        clip_eigs: bool = False,
        bidirectional: bool = False,
        step_rescale: float = 1.0,
        *, 
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.H, self.P, self.C_init = H, P, C_init
        self.discretization, self.dt_min, self.dt_max = discretization, dt_min, dt_max
        self.conj_sym, self.clip_eigs = conj_sym, clip_eigs
        self.bidirectional, self.step_rescale = bidirectional, step_rescale
        self.V, self.Vinv = V, Vinv

        if self.conj_sym:
                local_P = 2 * self.P
        else:
            local_P = self.P

        # Get RNG keys for parameter initializations
        key_B, key_C, key_C1, key_C2, key_D, key_log = jax.random.split(rngs.params(), 6)
        # Initialize trainable parameters
        self.Lambda_re = nnx.Param(Lambda_re_init)
        self.Lambda_im = nnx.Param(Lambda_im_init)
        B_init_fn = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = nnx.Param(init_VinvB(B_init_fn, key_B, B_shape, self.Vinv))
        if self.C_init == "trunc_standard_normal":
            C_init_fn = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init == "lecun_normal":
            C_init_fn = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init == "complex_normal":
            C_init_fn = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(f"C_init method {self.C_init} not implemented")
        if self.C_init == "complex_normal":
            c_shape = (self.H, 2 * self.P, 2) if self.bidirectional else (self.H, self.P, 2)
            self.C = nnx.Param(C_init_fn(key_C, c_shape))
        else:
            if self.bidirectional:
                self.C1 = nnx.Param(init_CV(C_init_fn, key_C1, C_shape, self.V))
                self.C2 = nnx.Param(init_CV(C_init_fn, key_C2, C_shape, self.V))
            else:
                self.C = nnx.Param(init_CV(C_init_fn, key_C, C_shape, self.V))
        
        self.D = nnx.Param(normal(stddev=1.0)(key_D, (self.H,)))
        
        self.log_step = nnx.Param(init_log_steps(key_log, (self.P, self.dt_min, self.dt_max)))
    
    def __call__(self, hidden, input_sequence, resets):
        # Recompute matrices on-the-fly from stored parameters
        if self.clip_eigs:
            Lambda = jnp.clip(self.Lambda_re.value, None, -1e-4) + 1j * self.Lambda_im.value
        else:
            Lambda = self.Lambda_re.value + 1j * self.Lambda_im.value

        B_tilde = self.B.value[..., 0] + 1j * self.B.value[..., 1]
        
        if self.C_init == "complex_normal":
            C_tilde = self.C.value[..., 0] + 1j * self.C.value[..., 1]
        else:
            if self.bidirectional:
                C1 = self.C1.value[..., 0] + 1j * self.C1.value[..., 1]
                C2 = self.C2.value[..., 0] + 1j * self.C2.value[..., 1]
                C_tilde = jnp.concatenate((C1, C2), axis=-1)
            else:
                C_tilde = self.C.value[..., 0] + 1j * self.C.value[..., 1]

        step = self.step_rescale * jnp.exp(self.log_step.value[:, 0])
        
        # Discretize
        if self.discretization == "zoh":
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif self.discretization == "bilinear":
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise NotImplementedError(f"Discretization method {self.discretization} not implemented")

        hidden, ys = apply_ssm(
            Lambda_bar, B_bar, C_tilde, hidden, input_sequence,
            resets, self.conj_sym, self.bidirectional
        )
        
        # Add feedthrough
        Du = jax.vmap(lambda u: self.D.value * u)(input_sequence)
        return hidden, ys + Du

def init_S5SSM(H, P, Lambda_re_init, Lambda_im_init, V, Vinv, C_init, discretization,
               dt_min, dt_max, conj_sym, clip_eigs, bidirectional, rngs):
    """Convenience function to partially initialize the S5SSM (NNX version)."""
    return partial(S5SSM, H=H, P=P, Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init, V=V, Vinv=Vinv, C_init=C_init,
                   discretization=discretization, dt_min=dt_min, dt_max=dt_max,
                   conj_sym=conj_sym, clip_eigs=clip_eigs, bidirectional=bidirectional, rngs=rngs)

def make_HiPPO(N):
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A

def make_NPLR_HiPPO(N):
    hippo = make_HiPPO(N)
    P = jnp.sqrt(jnp.arange(N) + 0.5)
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return hippo, P, B

def make_DPLR_HiPPO(N):
    A, P, B = make_NPLR_HiPPO(N)
    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]
    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)
    Lambda_imag, V = eigh(S * -1j)
    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig

class StackedEncoderModel(nnx.Module):
    """ Defines a stack of S5 layers to be used as an encoder (NNX Version)."""
    def __init__(
        self,
        ssm:  nnx.Module,
        d_model: int,
        n_layers: int,
        activation: str = "gelu",
        do_norm: bool = True,
        prenorm: bool = True,
        do_gtrxl_norm: bool = True,
        *,
        rngs: nnx.Rngs
    ):
        super().__init__()
        layer_keys = jax.random.split(rngs.params(), n_layers)
        self.layers = [
            SequenceLayer(
                ssm=ssm,
                d_model=d_model,
                activation=activation,
                do_norm=do_norm,
                prenorm=prenorm,
                do_gtrxl_norm=do_gtrxl_norm,
                rngs=nnx.Rngs(rng)
            )
            for rng in layer_keys
        ]

    def __call__(self, hidden, x, d):
        """
        Compute the LxH output of the stacked encoder.
        Args:
            hidden (list): A list of hidden state carries for each layer.
            x (float32): Input sequence (L, d_input).
            d (bool):    Reset signal (L,).
        Returns:
            A tuple with the list of new hidden states and the final output sequence.
        """
        new_hiddens = []
        for i, layer in enumerate(self.layers):
            new_h, x = layer(hidden[i], x, d)
            new_hiddens.append(new_h)
    
        return new_hiddens, x

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int, n_layers: int):
        # Use a dummy key since the default state init fn is just zeros.
        return [jnp.zeros((1, batch_size, hidden_size), dtype=jnp.float16) for _ in range(n_layers)]