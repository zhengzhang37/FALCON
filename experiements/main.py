import os
import sys
sys.path.append("../")
from pathlib import Path
import random
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.traverse_util import flatten_dict
import optax
from orbax import checkpoint as ocp
import grain.python as grain
import mlflow
from data_provider.dataloader import ImageDataSource
from utils.utils import (
    init_tx,
    initialize_dataloader,
)
import distrax
from typing import NamedTuple, Any, Sequence
from modules.cnn_s5 import ActorCriticRNN
from modules.s5 import StackedEncoderModel, make_DPLR_HiPPO, init_S5SSM
from modules.human_simulation import Human
import numpy as np
import time

def load_train_data_stream(
    data_source: ImageDataSource,
    clf_model: nnx.Module,
    cfg: DictConfig,
    ):
    dataloader_train = initialize_dataloader(
                data_source=data_source,
                num_epochs=1,
                shuffle=True,
                seed=random.randint(a=0, b=255),
                batch_size=cfg["training"]["num_steps"],
                crop_size=cfg["data_augmentation"]["crop_size"],
                resize=cfg["data_augmentation"]["resize"],
                padding_px=cfg["data_augmentation"]["padding_px"],
                mean=cfg["data_augmentation"]["mean"],
                std=cfg["data_augmentation"]["std"],
                p_flip=cfg["data_augmentation"]["prob_random_flip"],
                num_workers=1,
                num_threads=cfg["data_loading"]["num_threads"],
                prefetch_size=cfg["data_loading"]["prefetch_size"],
                is_color_img=True,
                drop_remainder=True
            )
    dataloader_train = iter(dataloader_train)
    xs, labels, logits = [], [], []
    for samples in tqdm(
        iterable= dataloader_train,
        total = cfg["dataset"]["length"]["train"] // cfg["training"]["num_steps"] + 1,
        desc='train',
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg["data_loading"]["progress_bar"]
        ):
        x = jnp.array(object=samples['image'], dtype=jnp.float32)
        y = jnp.array(object=samples['label'], dtype=jnp.int32)
        ai_logits = clf_model(x)
        labels.append(y)
        logits.append(ai_logits)
        xs.append(x)
    del dataloader_train
    labels = jnp.array(object=labels, dtype=jnp.int32)
    logits = jnp.array(object=logits, dtype=jnp.float32)
    xs = jnp.array(object=xs, dtype=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1)

    return {
        "images": xs,
        "labels": labels,
        "logits": logits,
    }

def load_test_data_stream(
    data_source: ImageDataSource,
    clf_model: nnx.Module,
    cfg: DictConfig,
    ):
    dataloader_test = initialize_dataloader(
                data_source=data_source,
                num_epochs=1,
                shuffle=False,
                seed=0,
                batch_size=cfg["testing"]["num_steps"],
                padding_px=None,
                crop_size=None,
                resize=cfg["data_augmentation"]["crop_size"],
                mean=cfg["data_augmentation"]["mean"],
                std=cfg["data_augmentation"]["std"],
                p_flip=None,
                is_color_img=True,
                num_workers=2,
                num_threads=cfg["data_loading"]["num_threads"],
                prefetch_size=cfg["data_loading"]["prefetch_size"],
            )
    # dataloader_test = iter(dataloader_test)
    labels, xs, logits = [], [], []
    for samples in tqdm(
        iterable= dataloader_test,
        total = cfg["dataset"]["length"]["test"] // cfg["testing"]["num_steps"],
        desc='test',
        ncols=80,
        leave=False,
        position=2,
        colour='blue',
        disable=not cfg["data_loading"]["progress_bar"]
        ):
        x = jnp.array(object=samples['image'], dtype=jnp.float32)
        y = jnp.array(object=samples['label'], dtype=jnp.int32)
        ai_logits = clf_model(x)
        labels.append(y)
        logits.append(ai_logits)
        xs.append(x)
    del dataloader_test
    labels = jnp.array(object=labels, dtype=jnp.int32)
    logits = jnp.array(object=logits, dtype=jnp.float32)
    xs = jnp.array(object=xs, dtype=jnp.float32)

    probs = jax.nn.softmax(logits, axis=-1)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1)

    return {
        "images": xs,
        "labels": labels,
        "logits": logits,
    }

def load_pretrained_model(model, checkpoint_path):
    ckptr = ocp.CheckpointManager(checkpoint_path)
    loaded_params = ckptr.restore(
        step=300,
        args=ocp.args.StandardRestore(item=nnx.state(model))
    )
    nnx.update(model, loaded_params)
    model.eval()
    return model


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    cost: jnp.ndarray
    cost_value: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    fatigue_ratio: jnp.ndarray
    info: jnp.ndarray

def calculate_gae(traj_batch, last_val, last_cost_val, cfg):
    """Calculates the Generalized Advantage Estimation for rewards and costs."""
    
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value, cost_gae, next_cost_value = gae_and_next_value
        done, value, reward, cost_value, cost = (
            transition.done,
            transition.value,
            transition.reward,
            transition.cost_value,
            transition.cost,
        )
        # Calculate deltas
        delta = reward + cfg["ppo"]["gamma"] * next_value * (1 - done) - value
        cost_delta = cost + cfg["ppo"]["gamma"] * next_cost_value * (1 - done) - cost_value
        
        # Update GAE
        gae = delta + cfg["ppo"]["gamma"] * cfg["ppo"]["gae_lambda"] * (1 - done) * gae
        cost_gae = cost_delta + cfg["ppo"]["gamma"] * cfg["ppo"]["gae_lambda"] * (1 - done) * cost_gae
        
        return (gae, value, cost_gae, cost_value), (gae, cost_gae)

    # Scan backwards through the trajectory
    _, (advantages, cost_advantages) = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val, jnp.zeros_like(last_cost_val), last_cost_val),
        traj_batch,
        reverse=True,
    )
    # Return advantages and targets for both reward and cost
    return advantages, advantages + traj_batch.value, cost_advantages, cost_advantages + traj_batch.cost_value


def make_train(cfg: DictConfig):

    cfg = OmegaConf.to_container(cfg, resolve=True)

    cfg["training"]["num_updates"] = (cfg["training"]["total_timesteps"] // cfg["training"]["num_steps"] // cfg["training"]["num_envs"])
    cfg["training"]["minibatch_size"] = cfg["training"]["num_envs"] * cfg["training"]["num_steps"] // cfg["training"]["num_minibatches"]
    def create_learning_rate_fn():
        base_learning_rate = cfg["training"]["lr"]
        lr_warmup = cfg["training"]["lr_warmup"]
        update_steps = cfg["training"]["num_updates"]
        warmup_steps = int(lr_warmup * update_steps)
        steps_per_epoch = (cfg["training"]["num_minibatches"] * cfg["training"]["update_epochs"])
        warmup_fn = optax.linear_schedule(init_value=0.0, end_value=base_learning_rate, transition_steps=warmup_steps * steps_per_epoch)
        cosine_epochs = max(update_steps - warmup_steps, 1)
        cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch)
        schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps * steps_per_epoch])
        return schedule_fn

    def train(rng, dataset_train, dataset_test, clf_model, ckpt_mngr):
        rng, ac_key, rnn_key = jax.random.split(rng, 3)
        d_model = cfg["s5"]["d_model"]
        ssm_size = cfg["s5"]["ssm_size"]
        n_layers = cfg["s5"]["n_layers"]
        blocks = cfg["s5"]["blocks"]
        block_size = int(ssm_size / blocks)

        Lambda, _, _, V,  _ = make_DPLR_HiPPO(ssm_size)
        block_size = block_size // 2
        ssm_size = ssm_size // 2
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vinv = V.conj().T

        ssm_init_fn = init_S5SSM(H=d_model,
            P=ssm_size,
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=Vinv,
            C_init="lecun_normal",
            discretization="zoh",
            dt_min=0.001,
            dt_max=0.1,
            conj_sym=True,
            clip_eigs=False,
            bidirectional=False,
        )
        ac_model = ActorCriticRNN(rngs=nnx.Rngs(ac_key), action_dim=2, ssm_init_fn=ssm_init_fn, config=cfg)
        tx = optax.chain(optax.clip_by_global_norm(cfg["ppo"]["max_grad_norm"]), optax.adam(learning_rate=create_learning_rate_fn(), eps=1e-5),)
        train_state = nnx.Optimizer(ac_model, tx)
        init_hstate = StackedEncoderModel.initialize_carry(
            batch_size=cfg["training"]["num_envs"], 
            hidden_size=ssm_size, 
            n_layers=cfg['s5']['n_layers']
        )

        init_hstate = jax.tree.map(lambda x: x.astype(jnp.complex64), init_hstate)

        lambda_lr = cfg["ppo"]["lambda_lr"]
        lagrange_lambda_lower = jnp.array(cfg["ppo"]["initial_lambda"], dtype=jnp.float32)
        lagrange_lambda_upper = jnp.array(cfg["ppo"]["initial_lambda"], dtype=jnp.float32)
        budget_upper = cfg["training"]["upper_bound"]
        budget_lower = cfg["training"]["lower_bound"]

        lambda_optimizer_lower = optax.adam(learning_rate=lambda_lr)
        lambda_opt_state_lower = lambda_optimizer_lower.init(lagrange_lambda_lower)

        lambda_optimizer_upper = optax.adam(learning_rate=lambda_lr)
        lambda_opt_state_upper = lambda_optimizer_upper.init(lagrange_lambda_upper)

        train_data = load_train_data_stream(
                data_source=dataset_train,
                clf_model=clf_model,
                cfg=cfg
        )

        test_obs_state = load_test_data_stream(
                data_source=dataset_test,
                clf_model=clf_model,
                cfg=cfg
        )
        @nnx.scan
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            train_state, update_step, hstate, lagrange_lambda_upper, lagrange_lambda_lower, lambda_opt_state_upper, lambda_opt_state_lower, rng = runner_state
            
            rng, shuffle_rng, human_rng = jax.random.split(rng, 3)
            shuffled_data = train_data
            num_envs = cfg["training"]["num_envs"]
            num_samples = train_data["images"].shape[0]
            num_envs = cfg["training"]["num_envs"]
            num_steps = cfg["training"]["num_steps"]
            permuted_indices = jax.random.permutation(shuffle_rng, num_samples)
            shuffled_data = jax.tree.map(lambda x: jnp.take(x, permuted_indices, axis=0), train_data)
            obs_state = jax.tree.map(lambda x: x[:num_envs], shuffled_data)

            @nnx.split_rngs(splits=cfg["training"]["num_envs"])
            @nnx.vmap(in_axes=0, out_axes=0)
            def create_human_model(rngs: nnx.Rngs):
                return Human(
                        num_classes=cfg["dataset"]["num_classes"], 
                        fatigue_params=cfg["human"]["fatigue_params"], 
                        sequence_length=cfg["training"]["num_steps"],
                        mimic_real_human=False,
                        rngs=rngs
                    )

            human_model = create_human_model(nnx.Rngs(human_rng))

            done = jnp.zeros((cfg["training"]["num_envs"], cfg["training"]["num_steps"]), dtype=bool).at[:, -1].set(True)
            human_fatigue = jnp.zeros((cfg["training"]["num_envs"], 1), dtype=jnp.float32)

            runner_state = (
                train_state,
                human_fatigue,
                update_step,
                hstate,
                rng,
            )
            @nnx.scan
            def _env_step(runner_state, step_idx):
                (
                    train_state,
                    human_fatigue,
                    update_step,
                    hstate,
                    rng,
                )  = runner_state

                # Get the current obs state
                rng, act_rng, human_rng = jax.random.split(rng, 3)
                image = obs_state['images'][:, step_idx]
                true_label = obs_state['labels'][:, step_idx]
                ai_logits = obs_state['logits'][:, step_idx]
                last_done = done[:, step_idx]
                obs = image
                fatigue_ratio = human_fatigue / cfg["training"]["num_steps"]
                
                # Agent acts
                hstate, pi, value, cost_value = train_state.model(hstate, (obs, last_done[np.newaxis, :], fatigue_ratio))
                action = pi.sample(seed=act_rng)
                log_prob = pi.log_prob(action)

                action = jnp.squeeze(action, axis=0)
                value = jnp.squeeze(value, axis=0)
                cost_value = jnp.squeeze(cost_value, axis=0)
                log_prob = jnp.squeeze(log_prob, axis=0)

                ai_pred = jnp.argmax(ai_logits, axis=-1)
                ai_pred_cross_entropy = -jnp.sum(jax.nn.one_hot(true_label, num_classes=cfg["dataset"]["num_classes"]) * jax.nn.log_softmax(ai_logits, axis=-1), axis=-1)
                human_pred = human_model(true_label, human_fatigue, rngs=nnx.Rngs(human_rng))
                human_pred = optax.smooth_labels(human_pred, alpha=0.01)
                human_pred = jnp.argmax(human_pred, axis=-1)
                final_pred = jnp.where(action == 1, human_pred, ai_pred)
                is_correct = (final_pred == true_label).astype(jnp.float32)
                human_performance = (human_pred == true_label).astype(jnp.float32)
                ai_performance = (ai_pred == true_label).astype(jnp.float32)
                human_fatigue = human_fatigue + action.squeeze()[:, np.newaxis]
                is_deferral = (action == 1).astype(jnp.float32)

                reward = is_correct
                cost = (action == 1).astype(jnp.float32)
                actual_reward = is_correct

                current_timestep = (
                    update_step * cfg["training"]["num_steps"] * cfg["training"]["num_envs"]
                )
                info = {}
                info["reward"] = reward
                info["cost"] = cost
                info["actual_reward"] = actual_reward
                info["human_fatigue"] = 1 - human_fatigue  / cfg["training"]["num_steps"]
                info["human_performance"] = human_performance
                info["ai_performance"] = ai_performance
                info["total_deferrals"] = is_deferral

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((cfg["training"]["num_envs"])), info
                )

                # Update human fatigue
                
                transition = Transition(
                    done=last_done, 
                    obs=obs, 
                    action=action, 
                    cost=cost,
                    cost_value=cost_value,
                    log_prob=log_prob, 
                    value=value, 
                    reward=reward,
                    fatigue_ratio=fatigue_ratio,
                    info=info
                    )

                runner_state = (
                    train_state,
                    human_fatigue,
                    update_step,
                    hstate,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = _env_step(
                runner_state, 
                jnp.arange(cfg["training"]["num_steps"])
            )

            
            train_state, human_fatigue, update_step, hstate, rng = runner_state
            initial_hstate = hstate
            _, _, last_val, last_cost_val = train_state.model(hstate, (traj_batch.obs[-1], done[:, -1][np.newaxis, :], traj_batch.fatigue_ratio[-1]))
            last_val = last_val.squeeze()
            last_cost_val = last_cost_val.squeeze()

            advantages, targets, cost_advantages, cost_targets = calculate_gae(traj_batch, last_val, last_cost_val, cfg)
            
            avg_cost = traj_batch.info["cost"].mean()

            violation_upper = budget_upper - avg_cost
            updates_upper, lambda_opt_state_upper = lambda_optimizer_upper.update(violation_upper, lambda_opt_state_upper, lagrange_lambda_upper)
            lagrange_lambda_upper = optax.apply_updates(lagrange_lambda_upper, updates_upper)
            lagrange_lambda_upper = jnp.maximum(0.0, lagrange_lambda_upper)

            violation_lower = budget_lower - avg_cost
            updates_lower, lambda_opt_state_lower = lambda_optimizer_lower.update(-violation_lower, lambda_opt_state_lower, lagrange_lambda_lower)
            lagrange_lambda_lower = optax.apply_updates(lagrange_lambda_lower, updates_lower)
            lagrange_lambda_lower = jnp.maximum(0.0, lagrange_lambda_lower)

            @nnx.scan
            def _update_epoch(update_state, unused):
                @nnx.scan
                def _update_minbatch(train_state, batch_info):
                    def _loss_fn(model, batch_info):
                        init_hstate, traj_batch, advantages, targets, cost_advantages, cost_targets = batch_info
                        _, pi, value, cost_value = model(init_hstate, (traj_batch.obs, traj_batch.done, traj_batch.fatigue_ratio))
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-cfg["ppo"]["clip_eps"], cfg["ppo"]["clip_eps"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE COST VALUE LOSS
                        cost_value_pred_clipped = traj_batch.cost + (
                            cost_value - traj_batch.cost
                        ).clip(-cfg["ppo"]["clip_eps"], cfg["ppo"]["clip_eps"])
                        cost_value_losses = jnp.square(cost_value - cost_targets)
                        cost_value_losses_clipped = jnp.square(cost_value_pred_clipped - cost_targets)
                        cost_value_loss = (
                            0.5 * jnp.maximum(cost_value_losses, cost_value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)

                        reward_gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                        cost_gae = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
                        penalty_multiplier = jax.lax.stop_gradient(lagrange_lambda_upper - lagrange_lambda_lower)
                        gae = reward_gae - penalty_multiplier * cost_gae
                        
                        loss_actor1 = ratio * gae
                        loss_actor2 = (jnp.clip(ratio, 1.0 - cfg["ppo"]["clip_eps"], 1.0 + cfg["ppo"]["clip_eps"],) * gae)
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + cfg["ppo"]["vf_coef"] * value_loss * 2
                            + cfg["ppo"]["vf_coef"] * cost_value_loss
                            - cfg["ppo"]["ent_coef"] * entropy
                        )
                        return total_loss, (value_loss, cost_value_loss, loss_actor, entropy)

                    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.model, batch_info)
                    train_state.update(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state

                rng, _rng = jax.random.split(rng)
                init_hstate = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (1, cfg["training"]["num_envs"], -1)), 
                    init_hstate
                )
                permutation = jax.random.permutation(_rng, cfg["training"]["num_envs"])
                batch = (
                    init_hstate, 
                    traj_batch, 
                    advantages, 
                    targets,
                    cost_advantages,
                    cost_targets
                    )
                # shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                # minibatches = jax.tree.map(
                #     lambda x: x.reshape(x.shape[0], cfg["training"]["num_minibatches"], -1, *x.shape[2:]).swapaxes(0, 1),
                #     shuffled_batch,
                # )

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1) if x.shape[0] > 1 else x,  # Only shuffle along batch dim
                    batch
                )
                minibatches = jax.tree.map(
                    lambda x: x.reshape(x.shape[0], cfg["training"]["num_minibatches"], -1, *x.shape[2:]).swapaxes(0, 1),
                    shuffled_batch,
                )

                train_state, losses = _update_minbatch(train_state, minibatches)
                init_hstate = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (1, cfg["training"]["num_envs"], -1)), 
                    init_hstate
                )

                
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state, 
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng
            )
            update_state, loss_info = _update_epoch(
                update_state, 
                jnp.arange(cfg["training"]["update_epochs"])
            )

            train_state = update_state[0]
            coverage = traj_batch.info["human_fatigue"][-1].mean()

            metrics = traj_batch.info
            rng = update_state[-1]
            
            update_step = update_step + 1
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
            metrics["update_step"] = update_step
            metrics["env_step"] = update_step * cfg["training"]["num_steps"] * cfg["training"]["num_envs"]
            mlflow.log_metrics(
                {
                    "loss": loss_info[0].mean(),
                    "value_loss": loss_info[1][0].mean(),
                    "actor_loss": loss_info[1][2].mean(),
                    "cost_loss": loss_info[1][1].mean(),
                    "lagrange_lambda_upper": lagrange_lambda_upper,
                    "lagrange_lambda_lower": lagrange_lambda_lower,
                    "entropy": loss_info[1][3].mean(),
                    "reward": metrics["reward"].mean(),
                    "actual_reward": metrics["actual_reward"].mean(),
                    "coverage": coverage,
                    "human_performance": traj_batch.info["human_performance"].mean(),
                },
                step=update_step
            )
            
            runner_state = (
                train_state,
                update_step,
                hstate,
                lagrange_lambda_upper,
                lagrange_lambda_lower,
                lambda_opt_state_upper,
                lambda_opt_state_lower,
                rng,
            )

            # ckpt_mngr.save(
            #     step=update_step,
            #     args=ocp.args.StandardSave(nnx.state(train_state.model))
            # )
            # ckpt_mngr.wait_until_finished()

            metrics = traj_batch.info
            metrics = jax.tree_util.tree_map(lambda x: x.mean(), metrics)
            metrics["update_step"] = update_step
            metrics["env_step"] = update_step * cfg["training"]["num_steps"] * cfg["training"]["num_envs"]
            
            # Add training metrics
            metrics["loss"] = loss_info[0].mean()
            metrics["value_loss"] = loss_info[1][0].mean()
            metrics["actor_loss"] = loss_info[1][2].mean()
            metrics["cost_loss"] = loss_info[1][1].mean()
            metrics["lagrange_lambda_upper"] = lagrange_lambda_upper
            metrics["lagrange_lambda_lower"] = lagrange_lambda_lower
            metrics["entropy"] = loss_info[1][3].mean()
            metrics["coverage"] = coverage
            metrics["human_performance"] = traj_batch.info["human_performance"].mean()

            def test(fatigue_params):
                rng = jax.random.PRNGKey(seed=123)
                rng, human_rng = jax.random.split(rng)
                @nnx.split_rngs(splits=cfg["testing"]["num_envs"])
                @nnx.vmap(in_axes=0, out_axes=0)
                def create_human_model(rngs: nnx.Rngs):
                    return Human(
                        num_classes=cfg["dataset"]["num_classes"], 
                        fatigue_params=fatigue_params, 
                        sequence_length=cfg["training"]["num_steps"],
                        mimic_real_human=True,
                        rngs=rngs
                    )
                test_human_model = create_human_model(nnx.Rngs(human_rng))

                # Initialize environment state
                rng, rnn_key = jax.random.split(rng)
                test_hstate = StackedEncoderModel.initialize_carry(
                    batch_size=cfg["testing"]["num_envs"], 
                    hidden_size=cfg['actor_critic']['hidden_dim'], 
                    n_layers=cfg['s5']['n_layers']
                )

                test_human_fatigue = jnp.zeros((cfg["testing"]["num_envs"], 1), dtype=jnp.float32)
                test_done = jnp.zeros((cfg["testing"]["num_envs"], cfg["testing"]["num_steps"]), dtype=bool)
                test_done = test_done.at[:, -1].set(True)

                def _test_env_step(runner_state, step_idx):
                    test_hstate, test_human_fatigue, rng = runner_state
                    # Get the current observation from the test set
                    image = test_obs_state['images'][:, step_idx]
                    true_label = test_obs_state['labels'][:, step_idx]
                    ai_logits = test_obs_state['logits'][:, step_idx]
                    last_done = test_done[:, step_idx]
                    obs = image
                    fatigue_ratio = test_human_fatigue / cfg["training"]["num_steps"]

                    # Agent acts deterministically
                    rng, act_rng, human_rng_step = jax.random.split(rng, 3)
                    test_hstate, pi, _, _ = ac_model(test_hstate, (obs, last_done[np.newaxis, :], fatigue_ratio))
                    action = pi.mode()
                    action = jnp.squeeze(action, axis=0)

                    # Simulate predictions
                    ai_pred = jnp.argmax(ai_logits, axis=-1)
                    human_pred = test_human_model(true_label, test_human_fatigue, rngs=nnx.Rngs(human_rng_step))
                    human_pred = jnp.argmax(human_pred, axis=-1)
                    final_pred = jnp.where(action == 1, human_pred, ai_pred)
                    
                    # Calculate metrics
                    is_correct = (final_pred == true_label).astype(jnp.float32)
                    
                    # Update human fatigue for the next step
                    test_human_fatigue = test_human_fatigue + action.squeeze()[:, np.newaxis]

                    test_metrics = {
                        "accuracy": is_correct,
                        "human_fatigue": (1 - test_human_fatigue.squeeze() / cfg["testing"]["num_steps"])
                    }
                    
                    return (test_hstate, test_human_fatigue, rng), test_metrics

                runner_state = (test_hstate, test_human_fatigue, rng)
                _, test_metrics = jax.lax.scan(
                    f=_test_env_step,
                    init=runner_state,
                    xs=jnp.arange(cfg["testing"]["num_steps"]),
                    length=cfg["testing"]["num_steps"],
                )

                return runner_state, test_metrics
            
            # Run the test function
            _, test_metrics = test(cfg["human"]["test_fatigue_params3"])

            mlflow.log_metrics(
                metrics={
                        'test_accuracy': test_metrics['accuracy'].mean(), 
                        'test_coverage': coverage.item(),
                        },
                step=update_step,
                synchronous=False
            )
            metrics["test_accuracy"] = test_metrics['accuracy']
            metrics["test_coverage"] = test_metrics['human_fatigue']

            return runner_state, metrics

                
        rng, rngs_ = jax.random.split(rng)

        runner_state = (
            train_state,
            0,
            init_hstate, 
            lagrange_lambda_upper,
            lagrange_lambda_lower,
            lambda_opt_state_upper,
            lambda_opt_state_lower,
            rngs_
            )
        runner_state, metrics = _update_step(runner_state, jnp.arange(cfg['training']['num_updates']))

        return {"runner_state": runner_state, "metrics": metrics}

        # for i in range(cfg['training']['num_updates']):
        #     step_metrics = {
        #         "loss": float(metrics["loss"][i]),
        #         "value_loss": float(metrics["value_loss"][i]),
        #         "actor_loss": float(metrics["actor_loss"][i]),
        #         "cost_loss": float(metrics["cost_loss"][i]),
        #         "lagrange_lambda_upper": float(metrics["lagrange_lambda_upper"][i]),
        #         "lagrange_lambda_lower": float(metrics["lagrange_lambda_lower"][i]),
        #         "entropy": float(metrics["entropy"][i]),
        #         "reward": float(metrics["reward"][i]),
        #         "actual_reward": float(metrics["actual_reward"][i]),
        #         "coverage": float(metrics["coverage"][i]),
        #         "human_performance": float(metrics["human_performance"][i]),
        #     }
            
        #     mlflow.log_metrics(step_metrics, step=int(i + 1))

    return train

@hydra.main(version_base=None, config_path="configs", config_name="falcon")
def main(cfg: DictConfig) -> None:
    """main procedure
    """
    # region ENVIRONMENT
    jax.config.update('jax_disable_jit', False)
    jax.config.update('jax_platforms', cfg.jax.platform)

    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(cfg.jax.mem)
    if not os.path.exists(path=cfg.experiment.logdir):
        Path(cfg.experiment.logdir).mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri=cfg.experiment.tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.experiment.name)
    mlflow.disable_system_metrics_logging()

    source_train = ImageDataSource(
        json_file=cfg.dataset.train_file,
        root=cfg.dataset.root
    )

    source_test = ImageDataSource(
        json_file=cfg.dataset.test_file,
        root=cfg.dataset.root
    )

    OmegaConf.set_struct(conf=cfg, value=True)
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.train',
        value=len(source_train),
        force_add=True
    )
    OmegaConf.update(
        cfg=cfg,
        key='dataset.length.test',
        value=len(source_test),
        force_add=True
    )
    with mlflow.start_run(run_id=cfg.experiment.run_id) as mlflow_run:
        ckpt_dir = os.path.join(os.getcwd(), cfg.experiment.logdir, cfg.experiment.name, mlflow_run.info.run_id)
        ckpt_options = ocp.CheckpointManagerOptions(save_interval_steps=10, max_to_keep=3)
        with ocp.CheckpointManager(directory=ckpt_dir, options=ckpt_options) as ckpt_mngr:
            start_epoch_id = 0
            if cfg.experiment.run_id is None:
                mlflow.log_params(params=flatten_dict(xs=OmegaConf.to_container(cfg=cfg), sep='.'))
                mlflow.log_artifact(local_path=os.path.abspath(path=__file__), artifact_path='source_code')
            else:
                start_epoch_id = ckpt_mngr.latest_step()

            OmegaConf.update(
                cfg=cfg,
                key='ckpt_dir',
                value=ckpt_dir,
                force_add=True
            )
            

            # define region AI CLF MODELS
            clf_model = hydra.utils.instantiate(config=cfg.model)(
                num_classes=cfg.dataset.num_classes,
                rngs=nnx.Rngs(jax.random.PRNGKey(seed=0)),
                # dropout_rate=cfg.training.dropout_rate,
                dtype=eval(cfg.jax.dtype)
            )
            clf_model = load_pretrained_model(
                model=clf_model, 
                checkpoint_path=cfg.pretrained.checkpoint
            )

            train_jit = make_train(cfg)   
            rng_key = jax.random.PRNGKey(0)
            out = train_jit(rng_key, source_train, source_test, clf_model, ckpt_mngr)
               

if __name__ == "__main__":
    main()