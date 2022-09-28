# https://www.mikulskibartosz.name/how-to-create-an-environment-for-a-tensorflow-agent/
# https://towardsdatascience.com/creating-a-custom-environment-for-tensorflow-agent-tic-tac-toe-example-b66902f73059

from enum import Enum
import tensorflow as tf
import numpy as np
from typing import Text, Optional

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

tf.compat.v1.enable_v2_behavior()


class TicTacToeBoardWithNoRulesEnvironment(py_environment.PyEnvironment):

  def __init__(self):
    super().__init__()
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=8, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(9,), dtype=np.int32, minimum=0, maximum=1, name='observation')
    self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    self._episode_ended = False

  def print_state(self):
      print(self._state)

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
      self._state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
      self._episode_ended = False
      return ts.restart(np.array(self._state, dtype=np.int32))

  def __is_spot_empty(self, index):
      return self._state[index] == 0

  def __all_spots_occupied(self):
      return all(item == 1 for item in self._state)

  def _step(self, action):
      if self._episode_ended:
          return self.reset()

      if self.__is_spot_empty(action):
          self._state[action] = 1

          if self.__all_spots_occupied():
              self._episode_ended = True
              return ts.termination(np.array(self._state, dtype=np.int32), 1)
          else:
              return ts.transition(np.array(self._state, dtype=np.int32), reward=0.1, discount=1.0)

      else:
          self._episode_ended = True
          return ts.termination(np.array(self._state, dtype=np.int32), -1)

  def render(self, mode: Text = 'rgb_array') -> Optional[types.NestedArray]:
      print(self._state)
      # dummy data, to be replace with real data
      arr = np.arange(30)
      return arr.reshape(2, 5, 3)


