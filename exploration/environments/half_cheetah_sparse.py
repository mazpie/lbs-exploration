import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}

def get_state_block(state, n_blocks):

    min_vel = -9.0
    max_vel = +9.0
    min_ang_vel = -18.0
    max_ang_vel = +18.0

    vel_range =  max_vel - min_vel
    vel_unit = vel_range / n_blocks
    ang_vel_range = max_ang_vel - min_ang_vel
    ang_vel_unit = ang_vel_range / n_blocks

    vel = state[0].item()
    ang_vel = state[1].item()

    vel_block = np.ceil( np.clip((vel - min_vel), 0, max_vel*2) / vel_unit)
    ang_vel_block = np.ceil( np.clip((ang_vel - min_ang_vel), 0, max_ang_vel*2) / ang_vel_unit)

    return [vel_block, ang_vel_block]

def rate_buffer_with_blocks(state_buffer, n_blocks=6):
    visited_blocks = [get_state_block(state, n_blocks=n_blocks) for state in state_buffer]
    uniques = np.unique(visited_blocks, axis=0)
    n_uniques = len(uniques)
    return n_uniques, uniques


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='half_cheetah.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self.pos_range = [0., 0.]
        self.vel_range = [0., 0.]
        self.angle_range = [0., 0.]

        self.x_velocity = 0.
        self.ang_vel_y = 0.

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)



    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        rooty_before = self.sim.data.qpos[2]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        rooty_after = self.sim.data.qpos[2]
        
        x_velocity = (x_position_after - x_position_before)/ self.dt

        ctrl_cost = self.control_cost(action)

        ang_vel_y = (rooty_after - rooty_before) / self.dt

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()

        # reward = forward_reward  if forward_reward >= 0 else 0. # - ctrl_cost
        reward = +1.0 if x_position_after > +10.0 else 0.0
        
        done = False

        if x_position_after < self.pos_range[0]:
            self.pos_range[0] = x_position_after
        if x_position_after > self.pos_range[1]:
            self.pos_range[1] = x_position_after
        if x_velocity < self.vel_range[0]:
            self.vel_range[0] = x_velocity
        if x_velocity > self.vel_range[1]:
            self.vel_range[1] = x_velocity
        if ang_vel_y < self.angle_range[0]:
            self.angle_range[0] = ang_vel_y
        if ang_vel_y > self.angle_range[1]:
            self.angle_range[1] = ang_vel_y

        self.x_velocity = x_velocity
        self.ang_vel_y = ang_vel_y


        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,

            "pos_range" : self.pos_range, 
            "vel_range" : self.vel_range,
            "angle_range" : self.angle_range,

            'obs': [x_velocity, ang_vel_y]
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def get_info(self):
        x_position_after = self.sim.data.qpos[0]
        return {
            'x_position': x_position_after,
            'x_velocity': self.x_velocity,

            "pos_range" : self.pos_range, 
            "vel_range" : self.vel_range,
            "angle_range" : self.angle_range,

            'obs': [self.x_velocity, self.ang_vel_y]
        }


    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)