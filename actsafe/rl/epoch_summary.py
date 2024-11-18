from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from actsafe.rl.trajectory import Trajectory


@dataclass
class EpochSummary:
    _data: list[list[Trajectory]] = field(default_factory=list)
    cost_boundary: float = 25.0

    @property
    def empty(self):
        return len(self._data) == 0

    @property
    def metrics(self) -> Tuple[float, float]:
        rewards, costs = [], []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                *_, r, c, _, _ = trajectory.as_numpy()
                rewards.append(r.sum())
                costs.append(c.sum())
        # Stack data from all tasks on the first axis,
        # giving a [batch_size * num_episodes, ] shape.
        stacked_rewards = np.stack(rewards)
        stacked_costs = np.stack(costs)
        return stacked_rewards.mean(), stacked_costs.mean()

    @property
    def videos(self):
        all_vids = []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                if len(trajectory.frames) > 0:
                    all_vids.append(trajectory.frames)
        if len(all_vids) == 0:
            return None
        vids = np.asarray(all_vids)[-1].transpose(1, 0, -1, 2, 3)
        return vids

    def extend(self, samples: List[Trajectory]) -> None:
        self._data.append(samples)
