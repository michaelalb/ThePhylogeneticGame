import torch.nn as nn
import torch.nn.functional as functional

from SharedConsts import HIDDEN_SIZES, DTYPE
from SharedConsts import IN_FEATURES


class Net(nn.Module):
	def __init__(self, in_features=IN_FEATURES, output=1):
		dtype = DTYPE
		h_sizes = [in_features] + HIDDEN_SIZES
		#   in_features: number of features of input.
		super(Net, self).__init__()
		# Hidden layers
		self.hidden = nn.ModuleList()
		for k in range(len(h_sizes) - 1):
			self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]).type(dtype))

		# Output layer
		self.out = nn.Linear(h_sizes[-1], output)

	def forward(self, x):
		# Feedforward
		for layer in self.hidden:
			x = functional.leaky_relu(layer(x))

		return self.out(x)


class BnNet(nn.Module):
	def __init__(self, in_features=IN_FEATURES, output=1):
		dtype = DTYPE
		h_sizes = [in_features] + HIDDEN_SIZES
		super().__init__()
		# Hidden layers
		self.hidden = nn.ModuleList()
		for k in range(len(h_sizes) - 1):
			self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]).type(dtype))
			self.hidden.append(nn.BatchNorm1d(h_sizes[k + 1]))
			self.hidden.append(nn.LeakyReLU())

		# Output layer
		self.out = nn.Linear(h_sizes[-1], output)

	def forward(self, x):
		# Feedforward
		for layer in self.hidden:
			x = layer(x)

		return self.out(x)
