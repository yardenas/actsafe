from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from omegaconf import DictConfig

from safe_opax.common.learner import Learner
from safe_opax.la_mbda import rssm
from safe_opax.la_mbda.augmented_lagrangian import AugmentedLagrangianPenalizer
from safe_opax.la_mbda.dummy_penalizer import DummyPenalizer
from safe_opax.la_mbda.lbsgd import LBSGDPenalizer
from safe_opax.la_mbda.replay_buffer import ReplayBuffer
from safe_opax.la_mbda.safe_actor_critic import SafeModelBasedActorCritic
from safe_opax.la_mbda.utils import marginalize_prediction
from safe_opax.la_mbda.world_model import WorldModel, evaluate_model, variational_step
from safe_opax.rl.epoch_summary import EpochSummary
from safe_opax.rl.metrics import MetricsMonitor
from safe_opax.rl.trajectory import TrajectoryData
from safe_opax.rl.types import FloatArray, Report
from safe_opax.rl.utils import Count, PRNGSequence, add_to_buffer
from safe_opax.la_mbda import cem


@eqx.filter_jit
def policy(model, prev_state, observation, key):
    config = cem.CEMConfig(num_particles=150, num_iters=10, num_elite=15, stop_cond=0.1)

    def sample(
        horizon,
        initial_state,
        key,
        policy,
    ):
        outs, _ = model.sample(horizon, initial_state, key, policy)
        return marginalize_prediction(outs)

    def per_env_policy(prev_state, observation, key):
        model_key, policy_key = jax.random.split(key)
        current_rssm_state = model.infer_state(
            prev_state.rssm_state, observation, prev_state.prev_action, model_key
        )
        action = cem.policy(
            jax.tree_map(
                lambda x: jnp.repeat(x[None], 5, 0), current_rssm_state
            ).flatten(),
            sample,
            15,
            jnp.zeros((15, 2)),
            policy_key,
            config,
        )[0]
        return action, AgentState(current_rssm_state, action)

    observation = preprocess(observation)
    return jax.vmap(per_env_policy)(
        prev_state, observation, jax.random.split(key, observation.shape[0])
    )


class AgentState(NamedTuple):
    rssm_state: rssm.State
    prev_action: jax.Array

    @classmethod
    def init(cls, batch_size: int, cells: rssm.RSSM, action_dim: int) -> "AgentState":
        rssm_state = cells.init()
        rssm_state = jax.tree_map(
            lambda x: jnp.repeat(x[None], batch_size, 0), rssm_state
        )
        prev_action = jnp.zeros((batch_size, action_dim))
        self = cls(rssm_state, prev_action)
        return self


def make_actor_critic(safe, state_dim, action_dim, cfg, key):
    # Account for the the discount factor in the budget.
    episode_safety_budget = (
        (
            (cfg.training.safety_budget / cfg.training.time_limit)
            / (1.0 - cfg.agent.safety_discount)
        )
        if cfg.agent.safety_discount < 1.0 - np.finfo(np.float32).eps
        else cfg.training.safety_budget
    ) + cfg.agent.safety_slack
    if safe:
        if cfg.agent.penalizer.name == "lbsgd":
            penalizer = LBSGDPenalizer(
                cfg.agent.penalizer.m_0,
                cfg.agent.penalizer.m_1,
                cfg.agent.penalizer.eta,
                cfg.agent.penalizer.eta_rate,
            )
        elif cfg.agent.penalizer.name == "lagrangian":
            penalizer = AugmentedLagrangianPenalizer(
                cfg.agent.penalizer.initial_lagrangian,
                cfg.agent.penalizer.initial_multiplier,
                cfg.agent.penalizer.multiplier_factor,
            )
        else:
            raise NotImplementedError
    else:
        penalizer = DummyPenalizer()
    return SafeModelBasedActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_config=cfg.agent.actor,
        critic_config=cfg.agent.critic,
        actor_optimizer_config=cfg.agent.actor_optimizer,
        critic_optimizer_config=cfg.agent.critic_optimizer,
        safety_critic_optimizer_config=cfg.agent.safety_critic_optimizer,
        horizon=cfg.agent.plan_horizon,
        discount=cfg.agent.discount,
        safety_discount=cfg.agent.safety_discount,
        lambda_=cfg.agent.lambda_,
        safety_budget=episode_safety_budget,
        penalizer=penalizer,
        key=key,
    )


class LaMBDA:
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        config: DictConfig,
    ):
        self.config = config
        self.replay_buffer = ReplayBuffer(
            observation_shape=observation_space.shape,
            action_shape=action_space.shape,
            max_length=config.training.time_limit // config.training.action_repeat,
            seed=config.training.seed,
            sequence_length=config.agent.replay_buffer.sequence_length
            // config.training.action_repeat,
            batch_size=config.agent.replay_buffer.batch_size,
            capacity=config.agent.replay_buffer.capacity,
        )
        self.prng = PRNGSequence(config.training.seed)
        action_shape = int(np.prod(action_space.shape))
        assert len(observation_space.shape) == 3
        self.model = WorldModel(
            image_shape=observation_space.shape,
            action_dim=action_shape,
            key=next(self.prng),
            **config.agent.model,
        )
        self.model_learner = Learner(self.model, config.agent.model_optimizer)
        self.state = AgentState.init(
            config.training.parallel_envs, self.model.cells, action_shape
        )
        self.should_train = Count(config.agent.train_every)
        self.metrics_monitor = MetricsMonitor()

    def __call__(
        self,
        observation: FloatArray,
        train: bool = False,
    ) -> FloatArray:
        if train and not self.replay_buffer.empty and self.should_train():
            self.update()
        actions, self.state = policy(
            self.model,
            self.state,
            observation,
            next(self.prng),
        )
        return np.asarray(actions)

    def observe(self, trajectory: TrajectoryData) -> None:
        add_to_buffer(
            self.replay_buffer,
            trajectory,
            self.config.training.scale_reward,
        )
        self.state = jax.tree_map(lambda x: jnp.zeros_like(x), self.state)

    def update(self):
        for batch in self.replay_buffer.sample(self.config.agent.update_steps):
            self.update_model(batch)

    def update_model(self, batch: TrajectoryData) -> jax.Array:
        features, actions = _prepare_features(batch)
        (self.model, self.model_learner.state), (loss, rest) = variational_step(
            features,
            actions,
            self.model,
            self.model_learner,
            self.model_learner.state,
            next(self.prng),
            self.config.agent.beta,
            self.config.agent.free_nats,
        )
        self.metrics_monitor["agent/model/loss"] = float(loss.mean())
        self.metrics_monitor["agent/model/reconstruction"] = float(
            rest["reconstruction_loss"].mean()
        )
        self.metrics_monitor["agent/model/kl"] = float(rest["kl_loss"].mean())
        return rest["states"].flatten()

    def report(self, summary: EpochSummary, epoch: int, step: int) -> Report:
        metrics = {
            k: float(v.result.mean) for k, v in self.metrics_monitor.metrics.items()
        }
        batch = next(self.replay_buffer.sample(1))
        features, actions = _prepare_features(batch)
        video = evaluate_model(self.model, features, actions, next(self.prng))
        return Report(metrics=metrics, videos={"agent/model/prediction": video})


@jax.jit
def _prepare_features(batch: TrajectoryData) -> tuple[rssm.Features, jax.Array]:
    reward = batch.reward[..., None]
    terminals = jnp.zeros_like(reward)
    features = rssm.Features(
        jnp.asarray(preprocess(batch.next_observation)),
        jnp.asarray(reward),
        jnp.asarray(batch.cost[..., None]),
        jnp.asarray(terminals),
    )
    actions = jnp.asarray(batch.action)
    return features, actions


def preprocess(image):
    return image / 255.0 - 0.5
