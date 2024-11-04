import numpy as np
from tqdm import tqdm

from actsafe.rl.episodic_async_env import EpisodicAsync
from actsafe.rl.epoch_summary import EpochSummary
from actsafe.rl.trajectory import Trajectory, Transition
from actsafe.rl.types import Agent


def interact(
    agent: Agent,
    environment: EpisodicAsync,
    num_steps: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[list[Trajectory], int]:
    observations = environment.reset()
    episodes: list[Trajectory] = []
    trajectories = [Trajectory() for _ in range(environment.num_envs)]
    track_rewards = np.zeros(environment.num_envs)
    track_costs = np.zeros(environment.num_envs)
    pbar = tqdm(
        range(0, num_steps, environment.action_repeat * environment.num_envs),
        unit=f"Steps (✕ {environment.num_envs} parallel)",
    )
    for _ in pbar:
        render = render_episodes > 0
        if render:
            images = environment.render()
            for i, trajectory in enumerate(trajectories):
                trajectory.frames.append(images[i])
        actions = agent(observations, train)
        next_observations, rewards, done, infos = environment.step(actions)
        costs = np.array([info.get("cost", 0) for info in infos])
        transition = Transition(
            observations, next_observations, actions, rewards, costs, done
        )
        for i, trajectory in enumerate(trajectories):
            trajectory.transitions.append(Transition(*map(lambda x: x[i], transition)))
        agent.observe_transition(transition)
        observations = next_observations
        step += environment.action_repeat
        track_rewards += rewards * (~done)
        track_costs += costs * (~done)
        pbar.set_postfix({"reward": track_rewards.mean(), "cost": track_costs.mean()})
        if render:
            render_episodes = max(render_episodes - done.any(), 0)
        for i, (ep_done, trajectory) in enumerate(zip(done, trajectories)):
            if ep_done:
                episodes.append(trajectory)
                trajectories[i] = Trajectory()
    return episodes, step


def epoch(
    agent: Agent,
    env: EpisodicAsync,
    num_steps: int,
    train: bool,
    step: int,
    render_episodes: int = 0,
) -> tuple[EpochSummary, int]:
    summary = EpochSummary()
    samples, step = interact(
        agent,
        env,
        num_steps,
        train,
        step,
        render_episodes,
    )
    summary.extend(samples)
    return summary, step
