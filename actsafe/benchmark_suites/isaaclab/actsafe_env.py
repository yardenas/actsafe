from gymnasium.spaces import Box
from omni.isaac.lab.envs.manager_based_rl_env import ManagerBasedRLEnv
import torch


class ManagerBasedActSafeEnv(ManagerBasedRLEnv):
    
    def step(self, actions):
        # process actions
        self.action_manager.process_action(actions.to(self.device))

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- final observations and extras
        self.final_obs_buf = self.observation_manager.compute()
        self.final_extras = self.extras
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()
        
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
        

class ActSafeEnvWrapper:
    
    def __init__(self, env: ManagerBasedActSafeEnv):
        self.env = env
        
    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return obs["policy"].cpu().numpy()
    
    def step(self, actions):
        actions = torch.tensor(actions, device=self.env.device, requires_grad=False)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        
        rew = rew.detach()
        extras["log"] = {k: v.detach() for k, v in extras["log"].items() if isinstance(v, torch.Tensor)}
        
        terminated, truncated = terminated.to(dtype=torch.bool), truncated.to(dtype=torch.bool)
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        if not self.env.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        
        infos = [
            {
                "log": {k: v.cpu().numpy() for k, v in extras["log"].items() if isinstance(v, torch.Tensor)},
                "observations": {"policy": extras["observations"]["policy"][i].cpu().numpy()},
                "time_outs": extras["time_outs"][i].cpu().numpy(),
                "final_observation": self.env.final_obs_buf["policy"][i].cpu().numpy(),
                "final_info": {
                    "log": {k: v.cpu().numpy() for k, v in self.env.final_extras["log"].items() if isinstance(v, torch.Tensor)},
                    "observations": {"policy": self.env.final_extras["observations"]["policy"][i].cpu().numpy()},
                    "time_outs": self.env.final_extras["time_outs"][i].cpu().numpy(),
                },
            }
            for i in range(self.env.num_envs)
        ]
        return obs.cpu().numpy(), rew.cpu().numpy(), terminated.cpu().numpy(), truncated.cpu().numpy(), infos

    @property
    def max_episode_length(self) -> int:
        return self.env.max_episode_length
    
    @property
    def num_envs(self) -> int:
        return self.env.num_envs
    
    @property
    def observation_space(self) -> Box:
        return self.env.single_observation_space["policy"]
    
    @property
    def action_space(self) -> Box:
        return self.env.single_action_space
    
    @property
    def action_repeat(self) -> int:
        return 1
