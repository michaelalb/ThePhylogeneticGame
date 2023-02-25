import random
from collections import deque

import joblib
import numpy as np
import torch

from SharedConsts import DTYPE
from SharedConsts import USE_CUDA


class ReplayBuffer:
	"""Fixed -size buffer to store experience tuples."""

	def __init__(self, buffer_size, batch_size, memory_path):
		"""
		Initialize a ReplayBuffer object.

		Params
		======
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
		"""

		if memory_path.is_file():
			with open(memory_path, 'rb') as fp:
				self.memory = joblib.load(fp)
			fp.close()
		else:
			self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experiences = None
		self.memory_path = memory_path

	def add(self, old_state_action, reward, done: bool, new_state_action):
		"""
		Add a new experience to memory.
		"""
		e = old_state_action.numpy(), np.array([reward]), np.array([int(done)]), new_state_action.numpy()

		# if buffer is full - removes oldest item
		self.memory.append(e)

	def sample(self):
		"""
		Randomly sample a batch of experiences from memory
		"""
		experiences = random.sample(self.memory, k=self.batch_size)  # returns a list of k tuples

		return self.process_experience(experiences)

	def process_experience(self, experiences):
		state_actions, rewards, dones, next_state_actions = map(np.array, zip(*experiences))
		# tensors
		state_actions = self.to_torch(state_actions)
		rewards = self.to_torch(rewards)
		next_state_actions = self.to_torch(next_state_actions)
		dones = self.to_torch(dones)

		return state_actions, rewards, dones, next_state_actions

	@staticmethod
	def to_torch(var):
		"""
		:param var: numpy array
		:return: the right kind of torch for us
		"""
		device = torch.device("cuda" if USE_CUDA else "cpu")
		dtype = DTYPE

		return torch.from_numpy(var).to(device=device, dtype=dtype)

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)

	def save_buffer(self):
		joblib.dump(self.memory, self.memory_path)


class GoodReplayBuffer(ReplayBuffer):
	"""
	Fixed -size buffer to store experience tuples.
	samples are maid from the experience gathered OR an existing good experience buffer.
	"""

	def __init__(self, buffer_size, batch_size, memory_path):
		"""
		Initialize a GoodReplayBuffer object.
		expects a file called "Replay_Buffer_memory_good.pkl"
		in memory_path folder
		"""

		super().__init__(buffer_size, batch_size, memory_path)
		good_buffer_file = memory_path.parent / "Replay_Buffer_memory_good.pkl"
		if good_buffer_file.is_file():
			with open(good_buffer_file, 'rb') as fp:
				self.good_memory = joblib.load(fp)
			fp.close()
		else:
			raise FileNotFoundError("expected {}".format(good_buffer_file))

	def sample(self):
		"""
		Randomly sample a batch of experiences from memory or from good memory
		good to normal ration is 1:1
		"""
		if random.random() < 0.5:
			return super(GoodReplayBuffer, self).sample()

		experiences = random.sample(self.good_memory, k=self.batch_size)  # returns a list of k tuples
		return self.process_experience(experiences)
