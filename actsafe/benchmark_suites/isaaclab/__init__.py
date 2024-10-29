from omegaconf import DictConfig


from actsafe.benchmark_suites.utils import get_domain_and_task
from actsafe.rl.types import EnvironmentFactory


def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        import gymnasium as gym

        _, task_cfg = get_domain_and_task(cfg)
        env = gym.make("Isaac-Velocity-Flat-Anymal-D-v0")
        return env

    return make_env
