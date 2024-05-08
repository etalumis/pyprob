import gymnasium.spaces
from .model import Model
from .state import sample, observe
from .util import InferenceNetwork, InferenceEngine
from .distributions import GymDiscrete, Normal

from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from PIL import Image

gymnasium.spaces.Discrete = GymDiscrete

import gymnasium as gym
import numpy as np

import warnings

def make_env(env_name, render_mode, max_episode_steps, **global_kwargs):
    specific_kwargs = {}
    if env_name == 'FrozenLake-v1':
        specific_kwargs = {
            'desc': global_kwargs.get('desc'),
            'map_name': global_kwargs.get('map_name', '4x4'),
            'is_slippery': global_kwargs.get('is_slippery', True)
        }
    #elif env_name == 'HalfCheetah-v3':
    #    specific_kwargs = {
    #        'xml_file': global_kwargs.get('xml_file', 'half_cheetah.xml')
    #    }
    else:
        warnings.warn(f"Creating environment with default settings as {env_name} is not explicitly handled.", UserWarning)
        specific_kwargs = global_kwargs  # Pass all global_kwargs to gym.make

    # Try to create the environment with specific or global kwargs
    try:
        env = gym.make(env_name, max_episode_steps = max_episode_steps, render_mode = render_mode, **specific_kwargs)
    except TypeError as e:
        warnings.warn(f"Failed to create {env_name} with the provided arguments due to: {str(e)}", UserWarning)
        # Optionally, try again with no kwargs or handle the error differently
        env = gym.make(env_name, render_mode)
    return env

class GymModel(Model):
    def __init__(self, name='Unnamed PyProb model', address_dict_file_name=None, env_name='FrozenLake-v1', render_mode="rgb_array", max_episode_steps=100, **kwargs):
        super().__init__(name, address_dict_file_name)
        defaultKwargs = {'desc': ["SFFG", "FHFH", "FFFH", "HFFF"], 
                         'map_name': "4x4",
                         'is_slippery': True}
        kwargs = {**defaultKwargs, **kwargs}
        self._env = make_env(env_name, render_mode, max_episode_steps, **kwargs)
        #self._env = TimeLimit(self._env, max_episode_steps=max_episode_steps)
        observation, info = self._env.reset()
        self.observation_noise = 0.0001
        self.render = False

    def forward(self):
        observation, info = self._env.reset()
        observations = [str(observation)]
        while True:
            if self.render:
                img = self._env.render()
                plt.imshow(img)
                plt.axis('off')  # Turn off axis numbering
                plt.show()
                clear_output(wait=True)
            action = sample(self._env.action_space, name = ', '.join(observations))  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = self._env.step(action.item())
            observations.append(str(observation))

            if terminated or truncated:
                observation, info = self._env.reset()
                observations = [str(observation)]
                observe(Normal(reward, self.observation_noise), name="reward")
                self._env.close()
                return reward

    def get_max_reward(self, max_runs = 1000):
        max_reward = -np.inf 
        for i in range(max_runs):
            observation, info = self._env.reset()
            while True:
               action = sample(self._env.action_space)  # agent policy that uses the observation and info
               observation, reward, terminated, truncated, info = self._env.step(action.item())  
               if terminated or truncated:
                    break
        return self._env.reward_range[-1]
            
    def inference_compilation(self, reward, training_traces = 20000, posterior_traces = 200, render = False, **kwards):
        self.render = render
        self.learn_inference_network(num_traces=training_traces,
                              observe_embeddings={'reward' : {'dim' : 32}},
                              inference_network=InferenceNetwork.FEEDFORWARD)
        return self.posterior_results(num_traces=posterior_traces, # the number of samples estimating the posterior
                                         inference_engine=InferenceEngine.IMPORTANCE_SAMPLING_WITH_INFERENCE_NETWORK, # specify which inference engine to use
                                         observe={'reward': reward} # assign values to the observed values
                                         )