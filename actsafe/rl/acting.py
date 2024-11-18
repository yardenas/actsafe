import numpy as np
from tqdm import tqdm

from actsafe.rl.episodic_async_env import EpisodicAsync
from actsafe.rl.epoch_summary import EpochSummary
from actsafe.rl.trajectory import Trajectory, Transition, TrajectoryData
from actsafe.rl.types import Agent


def interact(
    agent: Agent,
    environment: EpisodicAsync,
    num_steps: int,
    train: bool,
    render_episodes: int = 0,
) -> list[Trajectory]:
    observations = environment.reset()
    episodes: list[Trajectory] = []
    trajectories = [Trajectory() for _ in range(environment.num_envs)]
    track_rewards = np.zeros(environment.num_envs)
    track_costs = np.zeros(environment.num_envs)
    assert num_steps % (environment.action_repeat * environment.num_envs) == 0
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
        next_observations, rewards, terminated, truncated, infos = environment.step(
            actions
        )
        costs = np.array(get_costs(infos))
        transition = Transition(
            observations,
            next_observations,
            actions,
            rewards,
            costs,
            truncated,
            terminated,
        )
        done = terminated | truncated
        for i, trajectory in enumerate(trajectories):
            trajectory.transitions.append(Transition(*map(lambda x: x[i], transition)))
        agent.observe_transition(transition, infos)
        observations = next_observations
        track_rewards *= ~done
        track_rewards += rewards
        track_costs *= ~done
        track_costs += costs
        pbar.set_postfix({"reward": track_rewards.mean(), "cost": track_costs.mean()})
        if render:
            render_episodes = max(render_episodes - done.any(), 0)
        for i, (ep_done, trajectory, info) in enumerate(zip(done, trajectories, infos)):
            if ep_done:
                agent.observe(finalize_trajectory(trajectory, info), i)
                episodes.append(trajectory)
                trajectories[i] = Trajectory()
    return episodes


def finalize_trajectory(trajectory: Trajectory, info: dict) -> TrajectoryData:
    np_trajectory = trajectory.as_numpy()
    cost = np_trajectory.cost.copy()
    cost[-1] = info.get("final_info", {}).get("cost", 0.0)
    return TrajectoryData(
        np_trajectory.observation,
        np_trajectory.next_observation,
        np_trajectory.action,
        np_trajectory.reward,
        cost,
        np_trajectory.done,
        np_trajectory.terminal,
    )


def get_costs(infos):
    out = []
    for info in infos:
        if "final_info" in info:
            cost = info["final_info"].get("cost", 0)
        else:
            cost = info.get("cost", 0)
        out.append(cost)
    return out


def epoch(
    agent: Agent,
    env: EpisodicAsync,
    num_steps: int,
    train: bool,
    render_episodes: int = 0,
) -> EpochSummary:
    summary = EpochSummary()
    samples = interact(
        agent,
        env,
        num_steps,
        train,
        render_episodes,
    )
    summary.extend(samples)
    return summary
