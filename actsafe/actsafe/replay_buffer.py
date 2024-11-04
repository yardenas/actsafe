from typing import Iterator, Dict
import jax
import numpy as np

from actsafe.common.double_buffer import double_buffer
from actsafe.rl.trajectory import Transition, TrajectoryData


class ReplayBuffer:
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        max_length: int,
        seed: int,
        capacity: int,
        batch_size: int,
        sequence_length: int,
        num_rewards: int,
    ):
        self.episode_id = 0
        self.dtype = np.float32
        self.obs_dtype = np.uint8
        self.max_length = max_length
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.num_rewards = num_rewards

        # Main storage arrays
        self.observation = np.zeros(
            (capacity, max_length + 1) + observation_shape,
            dtype=self.obs_dtype,
        )
        self.action = np.zeros(
            (capacity, max_length) + action_shape,
            dtype=self.dtype,
        )
        self.reward = np.zeros(
            (capacity, max_length, num_rewards),
            dtype=self.dtype,
        )
        self.cost = np.zeros(
            (capacity, max_length),
            dtype=self.dtype,
        )
        self.terminated = np.ones(
            (capacity, max_length),
            dtype=bool,
        )
        self.episode_lengths = np.zeros(capacity, dtype=np.int32)

        # Tracking ongoing episodes
        self.ongoing_episodes: Dict[int, Dict] = {}

        self._valid_episodes = 0
        self.rs = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.capacity = capacity

    def _initialize_ongoing_episode(self, worker_id: int):
        """Initialize storage for a new ongoing episode."""
        return {
            "observation": np.zeros(
                (self.max_length + 1,) + self.observation_shape, dtype=self.obs_dtype
            ),
            "action": np.zeros(
                (self.max_length,) + self.action_shape, dtype=self.dtype
            ),
            "reward": np.zeros((self.max_length, self.num_rewards), dtype=self.dtype),
            "cost": np.zeros(self.max_length, dtype=self.dtype),
            "terminated": np.zeros(self.max_length, dtype=bool),
            "current_step": 0,
        }

    def _commit_episode(self, worker_id: int):
        """Commit a completed episode to the main buffer."""
        episode_data = self.ongoing_episodes[worker_id]
        current_step = episode_data["current_step"]

        if current_step == 0:  # Skip empty episodes
            return

        # Check if we've reached capacity
        if self.episode_id >= self.capacity:
            self.episode_id = 0

        # Copy data to main arrays
        self.observation[self.episode_id, : current_step + 1] = episode_data[
            "observation"
        ][: current_step + 1]
        self.action[self.episode_id, :current_step] = episode_data["action"][
            :current_step
        ]
        self.reward[self.episode_id, :current_step] = episode_data["reward"][
            :current_step
        ]
        self.cost[self.episode_id, :current_step] = episode_data["cost"][:current_step]
        self.terminated[self.episode_id, :current_step] = episode_data["terminated"][
            :current_step
        ]

        # Set episode length
        self.episode_lengths[self.episode_id] = current_step

        # Mark remaining timesteps as done
        self.terminated[self.episode_id, current_step:] = True

        # Increment counters
        self.episode_id += 1
        self._valid_episodes = min(self._valid_episodes + 1, self.capacity)

        # Clear the ongoing episode
        self.ongoing_episodes[worker_id] = self._initialize_ongoing_episode(worker_id)

    def add(self, step_data: Transition):
        """Add a single environment step to the buffer."""
        # Ensure reward has correct shape
        for i in range(step_data.reward.shape[0]):
            # Get worker ID for this step
            worker_id = i
            # Initialize ongoing episode if needed
            if worker_id not in self.ongoing_episodes:
                self.ongoing_episodes[worker_id] = self._initialize_ongoing_episode(
                    worker_id
                )

            episode_data = self.ongoing_episodes[worker_id]
            current_step = episode_data["current_step"]

            # Store current observation
            episode_data["observation"][current_step] = step_data.observation[i]
            episode_data["action"][current_step] = step_data.action[i]
            episode_data["reward"][current_step] = step_data.reward[i]
            episode_data["cost"][current_step] = step_data.cost[i]
            episode_data["terminated"][current_step] = step_data.done[i]

            # If episode terminated
            if step_data.done[i]:
                # Store final observation
                episode_data["observation"][
                    current_step + 1
                ] = step_data.next_observation[i]
                self._commit_episode(worker_id)
            else:
                # Continue episode
                episode_data["current_step"] = current_step + 1

                # Check if we've reached max length
                if current_step + 1 >= self.max_length:
                    episode_data["terminated"][current_step] = True
                    self._commit_episode(worker_id)

    def _sample_batch(
        self,
        batch_size: int,
        sequence_length: int,
        valid_episodes: int | None = None,
    ):
        if valid_episodes is not None:
            valid_episodes = valid_episodes
        else:
            valid_episodes = self._valid_episodes

        while True:
            episode_ids = self.rs.choice(valid_episodes, size=batch_size)
            low = np.array(
                [
                    self.rs.randint(
                        0, max(1, self.episode_lengths[episode_id] - sequence_length)
                    )
                    for episode_id in episode_ids
                ]
            )
            timestep_ids = low[:, None] + np.tile(
                np.arange(sequence_length + 1),
                (batch_size, 1),
            )
            for i, (episode_id, time_steps) in enumerate(
                zip(episode_ids, timestep_ids)
            ):
                episode_length = self.episode_lengths[episode_id]
                if time_steps[-1] >= episode_length:
                    # Adjust timesteps to end at episode termination
                    offset = time_steps[-1] - episode_length + 1
                    timestep_ids[i] -= offset

            a, r, c = [
                x[episode_ids[:, None], timestep_ids[:, :-1]]
                for x in (self.action, self.reward, self.cost)
            ]
            o = self.observation[episode_ids[:, None], timestep_ids]
            o, next_o = o[:, :-1], o[:, 1:]
            terminated = self.terminated[episode_ids[:, None], timestep_ids[:, :-1]]
            yield o, next_o, a, r, c, terminated, terminated

    def sample(self, n_batches: int) -> Iterator[TrajectoryData]:
        if self.empty:
            return
        iterator = (
            TrajectoryData(
                *next(self._sample_batch(self.batch_size, self.sequence_length))
            )
            for _ in range(n_batches)
        )
        if jax.default_backend() == "gpu":
            iterator = double_buffer(iterator)  # type: ignore
        yield from iterator

    @property
    def empty(self):
        return self._valid_episodes == 0
