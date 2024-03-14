import logging
import os
from typing import Callable, Optional

import cloudpickle
from omegaconf import DictConfig

from safe_opax.rl import acting, episodic_async_env
from safe_opax.rl.epoch_summary import EpochSummary
from safe_opax.rl.logging import StateWriter, TrainingLogger
from safe_opax.rl.types import Agent, EnvironmentFactory
from safe_opax.rl.utils import PRNGSequence

_LOG = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        make_env: EnvironmentFactory,
        agent: Agent,
        at_epoch: list[Callable[[EpochSummary, int, int, TrainingLogger], None]]
        | None = None,
        start_epoch: int = 0,
        step: int = 0,
        seeds: PRNGSequence | None = None,
    ):
        self.config = config
        self.agent = agent
        self.make_env = make_env
        self.epoch = start_epoch
        self.step = step
        self.seeds = seeds
        self.logger: Optional[TrainingLogger] = None
        self.state_writer: Optional[StateWriter] = None
        self.env: Optional[episodic_async_env.EpisodicAsync] = None
        self.at_epoch = at_epoch if at_epoch is not None else []

    def __enter__(self):
        log_path = os.getcwd()
        self.logger = TrainingLogger(self.config)
        self.state_writer = StateWriter(log_path)
        self.env = episodic_async_env.EpisodicAsync(
            self.make_env,
            self.config.training.parallel_envs,
            self.config.training.time_limit,
            self.config.training.action_repeat,
        )
        if self.seeds is None:
            self.seeds = PRNGSequence(self.config.training.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.logger is not None and self.state_writer is not None
        self.state_writer.write(self.state)
        self.state_writer.close()

    def train(self, epochs: Optional[int] = None) -> None:
        epoch, logger, state_writer = self.epoch, self.logger, self.state_writer
        assert logger is not None and state_writer is not None
        for epoch in range(epoch, epochs or self.config.training.epochs):
            _LOG.info(f"Training epoch #{epoch}")
            summary = self._run_training_epoch(
                episodes_per_epoch=self.config.training.episodes_per_epoch,
                prefix="train",
            )
            for at_epoch in self.at_epoch:
                at_epoch(summary, epoch, self.step, logger)
            self.epoch = epoch + 1
            state_writer.write(self.state)

    def _run_training_epoch(
        self,
        episodes_per_epoch: int,
        prefix: str,
    ) -> EpochSummary:
        agent, env, logger, seeds = self.agent, self.env, self.logger, self.seeds
        assert (
            env is not None
            and agent is not None
            and logger is not None
            and seeds is not None
        )
        env.reset(seed=int(next(seeds)[0].item()))
        summary, step = acting.epoch(
            agent,
            env,
            episodes_per_epoch,
            True,
            self.step,
        )
        objective, cost_rate, feasibilty = summary.metrics
        logger.log(
            {
                f"{prefix}/objective": objective,
                f"{prefix}/cost_rate": cost_rate,
                f"{prefix}/feasibility": feasibilty,
            },
            self.step,
        )
        self.step = step
        next(seeds)
        return summary

    @classmethod
    def from_pickle(cls, config: DictConfig, state_path: str) -> "Trainer":
        with open(state_path, "rb") as f:
            make_env, seeds, agent, epoch, step = cloudpickle.load(f).values()
        assert agent.config == config, "Loaded different hyperparameters."
        _LOG.info(f"Resuming from step {step}")
        return cls(
            config=agent.config,
            make_env=make_env,
            start_epoch=epoch,
            seeds=seeds,
            agent=agent,
            step=step,
        )

    @property
    def state(self):
        return {
            "make_env": self.make_env,
            "seeds": self.seeds,
            "agent": self.agent,
            "epoch": self.epoch,
            "step": self.step,
        }
