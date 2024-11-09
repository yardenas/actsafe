from omegaconf import DictConfig


from actsafe.benchmark_suites.utils import get_domain_and_task
from actsafe.rl.types import EnvironmentFactory


def make(cfg: DictConfig) -> EnvironmentFactory:
    import argparse
    from omni.isaac.lab.app import AppLauncher
    
    # add argparse arguments
    parser = argparse.ArgumentParser()
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()

    _, task_cfg = get_domain_and_task(cfg)

    args_cli.task = task_cfg.task
    args_cli.headless = task_cfg.headless

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch
    from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.anymal_d.flat_env_cfg import AnymalDFlatEnvCfg
    from actsafe.benchmark_suites.isaaclab.actsafe_env import ActSafeEnvWrapper

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    def make_env():
        import gymnasium as gym

        gym.register(
            id="Isaac-Velocity-Flat-Anymal-D-Actsafe-v0",
            entry_point="actsafe.benchmark_suites.isaaclab.actsafe_env:ManagerBasedActSafeEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": AnymalDFlatEnvCfg,
            },
        )

        env_cfg = AnymalDFlatEnvCfg()
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = ActSafeEnvWrapper(env)
        cfg.training.parallel_envs = env.num_envs
        cfg.training.steps_per_epoch = max(env.max_episode_length * env.num_envs, cfg.training.steps_per_epoch)
        return env

    return make_env
