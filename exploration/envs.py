import os

import gym
import numpy as np
import torch
from gym.spaces.box import Box

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ClipRewardEnv)
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

try:
    import retro
except ImportError:
    pass

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        aer = allow_early_resets
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif env_id == 'MarioBrosNoFrameskip-v4':
            env = retro.make('SuperMarioBros-Nes', 'Level1-1')
            buttons = env.buttons
            env = MarioXReward(env)
            env = FrameSkip(env, 4)
            env= ProcessFrame84(env, crop=True) 
            env = LimitedDiscreteActions(env, buttons)
            # frame stack by default
            aer = True
        elif env_id == 'MountainCarStochastic-Frozen':
            env = gym.make('MountainCarSparse-v0')
            env = StochasticEnv(env)
        elif env_id == 'MountainCarStochastic-Evolving':
            env = gym.make('MountainCarSparse-v0')
            env = StochasticEnv(env, force_zero_action=True)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = ProcessFrame84(env, crop=False)
            # frame stack by default
            env = ExtraTimeLimit(env, max_episode_steps=4500)
            if "Montezuma" in env_id:
                env = MontezumaInfoWrapper(env)
            env = AddRandomStateToInfo(env)
            aer = True


        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            print("Adding time limit...")
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=aer)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None):
    envs = [
        make_env(env_name, seed + i * 1000 , i, log_dir, allow_early_resets) # i * 1000, to ensure different processes have different seeds
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs) # , ret=False), deprecated in stable_baselines3
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()

# From the Large-Scale Curiosity Study repo (https://github.com/openai/large-scale-curiosity/)

import itertools
from copy import copy
from PIL import Image

class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps > self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()

class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    '''
    Reproduces the action space from curiosity paper.
    '''

    def __init__(self, env, all_buttons, whitelist=KNOWN_BUTTONS | KNOWN_SHOULDERS):
        gym.ActionWrapper.__init__(self, env)

        self._num_buttons = len(all_buttons)
        button_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_BUTTONS}
        buttons = [(), *zip(button_keys), *itertools.combinations(button_keys, 2)]
        shoulder_keys = {i for i in range(len(all_buttons)) if all_buttons[i] in whitelist & self.KNOWN_SHOULDERS}
        shoulders = [(), *zip(shoulder_keys), *itertools.permutations(shoulder_keys, 2)]
        arrows = [(), (4,), (5,), (6,), (7,)]  # (), up, down, left, right
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask

class MarioXReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.

    def reset(self):
        ob = self.env.reset()
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        levellow, levelhigh, xscrollHi, xscrollLo = \
            info["levelLo"], info["levelHi"], info["xscrollHi"], info["xscrollLo"]
        currentx = xscrollHi * 256 + xscrollLo
        new_level = [levellow, levelhigh]
        if new_level != self.current_level:
            self.current_level = new_level
            self.current_max_x = 0.
            reward = 0.
            self.visited_levels.add(tuple(self.current_level))
        else:
            if currentx > self.current_max_x:
                delta = currentx - self.current_max_x
                self.current_max_x = currentx
                reward = delta
            else:
                reward = 0.
        if done:
            info["levels"] = copy(self.visited_levels)
            info["retro_episode"] = dict(levels=copy(self.visited_levels))
        return ob, reward, done, info

class FrameSkip(gym.Wrapper):
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totrew = 0
        for _ in range(self.n):
            ob, rew, done, info = self.env.step(action)
            totrew += rew
            if done: break
        return ob, totrew, done, info

def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class MontezumaInfoWrapper(gym.Wrapper):
    ram_map = {
        "room": dict(
            index=3,
        ),
        "x": dict(
            index=42,
        ),
        "y": dict(
            index=43,
        ),
    }

    def __init__(self, env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.visited = set()
        self.visited_rooms = set()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        ram_state = unwrap(self.env).ale.getRAM()
        for name, properties in MontezumaInfoWrapper.ram_map.items():
            info[name] = ram_state[properties['index']]
        pos = (info['x'], info['y'], info['room'])
        self.visited.add(pos)
        self.visited_rooms.add(info["room"])
        if done:
            info['mz_episode'] = dict(pos_count=len(self.visited),
                                      visited_rooms=copy(self.visited_rooms))
            self.visited.clear()
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()

class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs, crop=self.crop)

    @staticmethod
    def process(frame, crop=True):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # mario resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        size = (84, 110 if crop else 84)
        resized_screen = np.array(Image.fromarray(img).resize(size,
                                                              resample=Image.BILINEAR), dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, d, info = self.env.step(action)
        if self.random_state_copy is not None:
            info['random_state'] = self.random_state_copy
            self.random_state_copy = None
        return ob, r, d, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.random_state_copy = copy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)

class StochasticEnv(gym.Wrapper):
    def __init__(self, env, random_dimensions=1, random_scale=1, force_zero_action=False):
        gym.Wrapper.__init__(self, env)
        
        self.random_dimensions = random_dimensions
        self.random_low = -random_scale * np.ones((self.random_dimensions))
        self.random_high = random_scale * np.ones((self.random_dimensions))
        self.rnd_state = np.zeros((self.random_dimensions))

        self.observation_space = Box(low=np.concatenate([self.observation_space.low, self.random_low]), high=np.concatenate([self.observation_space.high, self.random_high]), dtype=np.float32)
        self.action_space = Box(low=np.array([*self.action_space.low, -1]), high=np.array([*self.action_space.high, 1]), dtype=np.float32)
        
        self.last_obs = self.observation_space.sample()
        self.last_done = False
        self.last_rew = 0.
        self.last_info = self.env.get_info()
        self.force_zero_action = force_zero_action

    def reset(self):
        ob = self.env.reset()
        self.rnd_state = np.zeros((self.random_dimensions))

        self.last_obs = np.concatenate([ob, self.rnd_state])
        self.last_rew = 0.
        self.last_done = False
        self.last_info = self.env.get_info()
        
        return self.last_obs

    def step(self, action):
        if action[-1] > 0:
            self.rnd_state = self.np_random.uniform(low=self.random_low, high=self.random_high)
            if self.force_zero_action:
                a = np.zeros_like(action)
                ob, reward, done, info = self.env.step(a[:-1])
                self.last_rew = reward
                self.last_done = done
                self.last_info = info
            else:
                ob = self.last_obs[:-self.random_dimensions]
                self.last_rew = 0
                self.last_done = False
                self.last_info = self.env.get_info()
                if isinstance(self.env, gym.wrappers.TimeLimit):
                    self.env._elapsed_steps += 1
                    if self.env._elapsed_steps >= self.env._max_episode_steps:
                        self.last_info["TimeLimit.truncated"] = not self.last_done
                        self.last_done = True
        else:
            ob, reward, done, info = self.env.step(action[:-1])
            self.last_rew = reward
            self.last_done = done
            self.last_info = info
        
        self.last_obs = np.concatenate([ob, self.rnd_state])
        return self.last_obs, self.last_rew, self.last_done, self.last_info