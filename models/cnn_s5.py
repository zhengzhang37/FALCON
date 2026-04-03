import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
import distrax
from flax import nnx
from typing import Sequence, NamedTuple, Any, Dict, Tuple
from models.s5 import StackedEncoderModel
from orbax import checkpoint as ocp
import hydra
from collections.abc import Sequence, Callable

class BasicBlock(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            dropout_rate: float,
            in_planes: int,
            planes: int,
            stride: int = 1,
            expansion: int = 1,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

        if stride != 1 or in_planes != (expansion * planes):
            self.shortcut = nnx.Conv(
                in_features=in_planes,
                out_features=expansion * planes,
                kernel_size=(1, 1),
                strides=stride,
                use_bias=False,
                dtype=dtype,
                rngs=rngs
            )
        else:
            self.shortcut = lambda x: x

        self.conv1 = nnx.Conv(
            in_features=in_planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(
            num_features=planes,
            dtype=dtype,
            rngs=rngs
        )
        
        self.conv2 = nnx.Conv(
            in_features=planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(
            num_features=planes,
            dtype=dtype,
            rngs=rngs
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv1(inputs=x)
        out = self.bn1(x=out)
        out = nnx.relu(x=out)

        out = self.dropout(out)

        out = self.conv2(inputs=out)
        out = self.bn2(x=out)

        out = out + self.shortcut(x)
        out = nnx.relu(x=out)

        return out


class Bottleneck(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            dropout_rate: float,
            in_planes: int,
            planes: int,
            stride: int = 1,
            expansion: int = 4,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

        if stride != 1 or in_planes != (expansion * planes):
            self.shortcut = nnx.Conv(
                in_features=in_planes,
                out_features=expansion * planes,
                kernel_size=(1, 1),
                strides=stride,
                use_bias=False,
                dtype=dtype,
                rngs=rngs
            )
        else:
            self.shortcut = lambda x: x

        self.conv1 = nnx.Conv(
            in_features=in_planes,
            out_features=planes,
            kernel_size=(1, 1),
            strides=stride,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=planes, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=self.bn1.num_features,
            out_features=self.bn1.num_features,
            kernel_size=(3, 3),
            strides=stride,
            dtype=dtype,
            rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(
            num_features=planes,
            dtype=dtype,
            rngs=rngs
        )
        self.conv3 = nnx.Conv(
            in_features=planes,
            out_features=planes * self.expansion,
            kernel_size=(1, 1),
            dtype=dtype,
            rngs=rngs
        )
        self.bn3 = nnx.BatchNorm(
            num_features=planes * self.expansion,
            dtype=dtype,
            rngs=rngs
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)

    def forward(self, x: jax.Array) -> jax.Array:
        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nnx.relu(out)

        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.shortcut(x)

        out = nnx.relu(out + shortcut)

        return out


class PreActBlock(nnx.Module):
    """Implement the pre-activated residual neural networks in the paper:
    'Identity Mappings in Deep Residual Networks' (ECCV 2016)
    """
    def __init__(
            self,
            rngs: nnx.Rngs,
            dropout_rate: float,
            in_planes: int,
            planes: int,
            stride: int = 1,
            expansion: int = 1,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

        self.bn1 = nnx.BatchNorm(
            num_features=in_planes,
            dtype=dtype,
            rngs=rngs
        )
        self.conv1 = nnx.Conv(
            in_features=in_planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )

        self.bn2 = nnx.BatchNorm(
            num_features=planes,
            dtype=dtype,
            rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )

        if stride != 1 or in_planes != (expansion * planes):
            self.shortcut = nnx.Conv(
                in_features=in_planes,
                out_features=expansion * planes,
                kernel_size=(1, 1),
                strides=stride,
                use_bias=False,
                dtype=dtype,
                rngs=rngs
            )
        else:
            self.shortcut = lambda x: x
        
        self.dropout = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.bn1(x=x)
        out = nnx.relu(x=out)

        shortcut = self.shortcut(x)

        out = self.conv1(inputs=out)

        out = self.bn2(x=out)
        out = nnx.relu(x=out)

        out = self.dropout(out)

        out = self.conv2(inputs=out)

        return out + shortcut


class PreActBottleneck(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            dropout_rate: float,
            in_planes: int,  # number of input channels
            planes: int,  # number of output channels
            stride: int,
            expansion: int = 4,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()
        self.expansion = expansion

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nnx.Conv(
                in_features=in_planes,
                out_features=expansion * planes,
                kernel_size=(1, 1),
                strides=stride,
                use_bias=False,
                dtype=dtype,
                rngs=rngs
            )
        else:
            self.shortcut = lambda x: x
        
        self.bn1 = nnx.BatchNorm(num_features=in_planes, rngs=rngs)

        self.conv1 = nnx.Conv(
            in_features=in_planes,
            out_features=planes,
            kernel_size=(1, 1),
            strides=1,
            padding=0,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )

        self.bn2 = nnx.BatchNorm(num_features=planes, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=planes,
            out_features=planes,
            kernel_size=(3, 3),
            strides=stride,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )

        self.bn3 = nnx.BatchNorm(num_features=planes, rngs=rngs)

        self.conv3 = nnx.Conv(
            in_features=planes,
            out_features=expansion * planes,
            kernel_size=(1, 1),
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )

        self.dropout = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)


    def __call__(self, x: jax.Array) -> jax.Array:
        shortcut = self.shortcut(x)

        out = self.bn1(x=x)
        out = nnx.relu(x=out)

        out = self.conv1(inputs=out)

        out = self.bn2(x=out)
        out = nnx.relu(x=out)

        out = self.conv2(inputs=out)

        out = self.bn3(x=out)
        out = nnx.relu(x=out)

        out = self.dropout(out)

        out = self.conv3(inputs=out)

        out = out + shortcut

        return out


class ResNet(nnx.Module):
    def __init__(
            self,
            in_channels: int,
            conv1_kernel_size: tuple[int, int] | int,
            rngs: nnx.Rngs,
            dropout_rate: float,
            block: Callable[..., BasicBlock | Bottleneck],
            num_blocks: Sequence[int],
            in_planes: int = 64,
            num_classes: int | None = None,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()

        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=64,
            kernel_size=conv1_kernel_size,
            strides=2,
            padding=3,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=in_planes, dtype=dtype, rngs=rngs)

        planes = [64, 128, 256, 512]
        self.layers = []
        for i in range(len(num_blocks)):
            in_planes, layer_temp = make_layer(
                block=block,
                in_planes=in_planes,
                planes=planes[i],
                num_blocks=num_blocks[i],
                stride=min(i + 1, 2),
                rngs=rngs,
                dropout_rate=dropout_rate,
                dtype=dtype
            )

            self.layers.append(layer_temp)

        if num_classes is None:
            self.clf = lambda x: x
        else:
            self.clf = nnx.Linear(
                in_features=int(512 * in_planes / planes[-1]),
                out_features=num_classes,
                dtype=dtype,
                rngs=rngs
            )
        
        # dropout of the first conv and the last fully-connected layers
        self.dropout_conv = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)
        self.dropout_mlp = nnx.Dropout(rate=dropout_rate, broadcast_dims=(-1,), rngs=rngs)

    def get_features(self, x: jax.Array) -> jax.Array:
        out = self.conv1(inputs=x)
        out = self.bn1(x=out)
        out = nnx.relu(x=out)
        out = nnx.max_pool(inputs=out, window_shape=(3, 3), strides=(2, 2), padding='VALID')

        out = self.dropout_conv(out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)

        out = jnp.mean(a=out, axis=(1, 2))

        return out


    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.get_features(x=x)

        out_features = self.dropout_mlp(out)

        out = self.clf(out_features)

        return out_features


class PreActResNet(nnx.Module):
    def __init__(
            self,
            in_channels: int,
            conv1_kernel_size: tuple[int, int] | int,
            rngs: nnx.Rngs,
            dropout_rate: float,
            block: Callable[..., PreActBlock | PreActBottleneck],
            num_blocks: Sequence[int],
            in_planes: int = 64,
            num_classes: int | None = None,
            dtype: jnp.dtype = jnp.float32) -> None:
        super().__init__()

        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=64,
            kernel_size=conv1_kernel_size,
            strides=1,
            padding=1,
            use_bias=False,
            dtype=dtype,
            rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_features=in_planes, dtype=dtype, rngs=rngs)

        planes = [64, 128, 256, 512]
        self.layers = []
        for i in range(len(num_blocks)):
            in_planes, layer_temp = make_layer(
                block=block,
                in_planes=in_planes,
                planes=planes[i],
                num_blocks=num_blocks[i],
                stride=min(i + 1, 2),
                rngs=rngs,
                dropout_rate=dropout_rate,
                dtype=dtype
            )
            self.layers.append(layer_temp)

        # dropout of the first conv and the last fully-connected layers
        self.dropout_conv = nnx.Dropout(rate=dropout_rate, broadcast_dims=(1, 2), rngs=rngs)
        self.dropout_mlp = nnx.Dropout(rate=dropout_rate, broadcast_dims=(-1,), rngs=rngs)
        
        if num_classes is None:
            self.clf = lambda x: x
        else:
            self.clf = nnx.Linear(
                in_features=int(512 * in_planes / planes[-1]),
                out_features=num_classes,
                dtype=dtype,
                rngs=rngs
            )

    def get_features(self, x: jax.Array) -> jax.Array:
        out = self.conv1(inputs=x)

        out = self.dropout_conv(out)

        for i in range(len(self.layers)):
            out = self.layers[i](out)

        out = jnp.mean(a=out, axis=(1, 2))

        return out

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.get_features(x=x)

        out_features = self.dropout_mlp(out)

        out = self.clf(out_features)

        return out_features


def ResNet18(
        num_classes: int,
        in_channels: int,
        conv1_kernel_size: tuple[int, int] | int,
        rngs: nnx.Rngs,
        dropout_rate: float,
        dtype: jnp.dtype = jnp.float32) -> ResNet:
    return ResNet(
        in_channels=in_channels,
        conv1_kernel_size=conv1_kernel_size,
        rngs=rngs,
        dropout_rate=dropout_rate,
        block=BasicBlock,
        num_blocks=(2, 2, 2, 2),
        num_classes=num_classes,
        dtype=dtype
    )

def PreActResNet18(
        num_classes: int,
        in_channels: int,
        conv1_kernel_size: tuple[int, int] | int,
        rngs: nnx.Rngs,
        dropout_rate: float,
        dtype: jnp.dtype = jnp.float32) -> PreActResNet:
    return PreActResNet(
        in_channels=in_channels,
        conv1_kernel_size=conv1_kernel_size,
        rngs=rngs,
        dropout_rate=dropout_rate,
        block=PreActBlock,
        num_blocks=(2, 2, 2, 2),
        num_classes=num_classes,
        dtype=dtype
    )

def make_layer(
        block: Callable[..., BasicBlock | Bottleneck | PreActBlock | PreActBottleneck],
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        rngs: nnx.Rngs,
        dropout_rate: float,
        dtype: jnp.dtype = jnp.float32) -> tuple[int, nnx.Module]:
    """
    """
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        block_layer = block(
            rngs=rngs,
            dropout_rate=dropout_rate,
            in_planes=in_planes,
            planes=planes,
            stride=stride,
            dtype=dtype
        )
        layers.append(block_layer)
        in_planes = planes * block_layer.expansion

    return in_planes, nnx.Sequential(*layers)

def load_pretrained_model(model, checkpoint_path):
    ckptr = ocp.CheckpointManager(checkpoint_path)
    loaded_params = ckptr.restore(
        step=300,
        args=ocp.args.StandardRestore(item=nnx.state(model))
    )
    nnx.update(model, loaded_params)
    return model

class ActorCriticRNN(nnx.Module):
    def __init__(
        self, 
        rngs: nnx.Rngs,
        action_dim: int, 
        ssm_init_fn: nnx.Module,
        config: Dict
        ):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.ssm_init_fn = ssm_init_fn
        self.cnn = ResNet18(
            num_classes=config["dataset"]["num_classes"],
            in_channels=config["model"]["in_channels"],
            conv1_kernel_size=config["model"]["conv1_kernel_size"],
            dropout_rate=config["model"]["dropout_rate"],
            rngs=nnx.Rngs(jax.random.PRNGKey(seed=0)),
            dtype=jnp.float32
        )
            
        
        self.cnn = load_pretrained_model(self.cnn, config["pretrained"]["checkpoint"])

        self.rnn = StackedEncoderModel(
            ssm=self.ssm_init_fn,
            d_model=self.config["s5"]["d_model"],
            n_layers=self.config["s5"]["n_layers"],
            activation=self.config["s5"]["activation"],
            do_norm=self.config["s5"]["do_norm"],
            prenorm=self.config["s5"]["prenorm"],
            do_gtrxl_norm=self.config["s5"]["do_gtrxl_norm"],
            rngs=rngs,
        )

        self.fatigue_encoder1 = nnx.Linear(in_features=1, out_features=32, rngs=rngs)
        self.fatigue_encoder2 = nnx.Linear(in_features=32, out_features=32, rngs=rngs)

        self.obs_encoder1 = nnx.Linear(in_features=config["actor_critic"]["hidden_dim"] + 32, out_features=config["actor_critic"]["hidden_dim"], rngs=rngs)
        self.obs_encoder2 = nnx.Linear(in_features=config["actor_critic"]["hidden_dim"], out_features=config["actor_critic"]["hidden_dim"], rngs=rngs)

        # --- Actor ---
        self.actor_linear1 = nnx.Linear(
            in_features=config["actor_critic"]["hidden_dim"],
            out_features=config["actor_critic"]["hidden_dim"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            rngs=rngs
        )

        self.actor_linear2 = nnx.Linear(
            in_features=config["actor_critic"]["hidden_dim"],
            out_features=self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs
        )

        # --- Critic ---
        self.critic_linear1 = nnx.Linear(
            in_features=config["actor_critic"]["hidden_dim"],
            out_features=config["actor_critic"]["fc_dim_size"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            rngs=rngs
        )
        self.critic_linear2 = nnx.Linear(
            in_features=config["actor_critic"]["fc_dim_size"],
            out_features=1,
            kernel_init=orthogonal(2),
            bias_init=constant(1.0),
            rngs=rngs
        )

        # --- Cost ---
        self.cost_linear1 = nnx.Linear(
            in_features=config["actor_critic"]["hidden_dim"],
            out_features=config["actor_critic"]["fc_dim_size"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            rngs=rngs
        )
        self.cost_linear2 = nnx.Linear(
            in_features=config["actor_critic"]["fc_dim_size"],
            out_features=1,
            kernel_init=orthogonal(2),
            bias_init=constant(1.0),
            rngs=rngs
        )

    
    def __call__(
        self, 
        hidden: jax.Array, 
        x: jax.Array
        )-> tuple[jax.Array, distrax.Distribution, jax.Array]:
        obs, dones, fatigue = x
        if len(obs.shape) > 4: 
            seq_len, batch_size = obs.shape[0], obs.shape[1]
            fatigue_embedding = self.fatigue_encoder1(fatigue.reshape(-1, *fatigue.shape[2:]))
            fatigue_embedding = nnx.relu(x=fatigue_embedding)
            fatigue_embedding = self.fatigue_encoder2(fatigue_embedding)
            embedding = self.cnn(obs.reshape(-1, *obs.shape[2:]))
            embedding = jnp.concatenate((embedding, fatigue_embedding), axis=-1)
            embedding = self.obs_encoder1(embedding)
            embedding = nnx.relu(x=embedding)
            embedding = self.obs_encoder2(embedding)
            embedding = embedding.reshape(seq_len, batch_size, -1) 
        else: 
            fatigue_embedding = self.fatigue_encoder1(fatigue)
            fatigue_embedding = nnx.relu(x=fatigue_embedding)
            fatigue_embedding = self.fatigue_encoder2(fatigue_embedding)
            embedding = self.cnn(obs)
            embedding = jnp.concatenate((embedding, fatigue_embedding), axis=-1)
            embedding = self.obs_encoder1(embedding)
            embedding = nnx.relu(x=embedding)
            embedding = self.obs_encoder2(embedding)

            embedding = embedding[jnp.newaxis, :, :] 
            if len(dones.shape) == 1:
                dones = dones[jnp.newaxis, :]  

        hidden, embedding = self.rnn(hidden, embedding, dones)
 
        # --- Actor ---
        actor_mean = self.actor_linear1(embedding)
        actor_mean = nnx.relu(x=actor_mean)
        actor_mean = self.actor_linear2(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        # --- Critic ---
        critic = self.critic_linear1(embedding)
        critic = nnx.relu(x=critic)
        critic = self.critic_linear2(critic)

        # --- Cost ---
        cost = self.cost_linear1(embedding)
        cost = nnx.relu(x=cost)
        cost = self.cost_linear2(cost)
        return hidden, pi, jnp.squeeze(critic, axis=-1), jnp.squeeze(cost, axis=-1)
