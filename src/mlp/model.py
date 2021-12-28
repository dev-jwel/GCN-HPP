import torch
import numpy as np
from torch.autograd import Variable
from loguru import logger

class Model(torch.nn.Module):
	def __init__(self, num_features, device):
		super(Model, self).__init__()

		self.norms = torch.nn.ModuleList()
		self.layers = torch.nn.ModuleList()
		self.activation = torch.nn.Tanh()

		# we fix last demension to 1 because our task is a regression
		for in_dim, out_dim in zip(num_features, num_features[1:] + [1]):
			self.norms.append(torch.nn.LayerNorm(in_dim).to(device))
			self.layers.append(torch.nn.Linear(in_dim, out_dim).to(device))

	def forward(self, feature, mask):
		out = feature * mask
		for norm, layer in zip(self.norms[:-1], self.layers[:-1]):
			out = norm(out) * mask
			out = layer(out) * mask
			out = self.activation(out) * mask
		out = self.norms[-1](out) * mask
		return self.layers[-1](out) * mask
