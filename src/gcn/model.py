import torch
from torch.autograd import Variable
from loguru import logger

class Layer(torch.nn.Module):
	def __init__(self, in_dim, out_dim, device):
		super(Layer, self).__init__()

		self.weight = torch.nn.Parameter(torch.Tensor(in_dim, out_dim).to(device))
		self.weight = torch.nn.init.xavier_uniform_(self.weight)
		self.bias = torch.nn.Parameter(torch.zeros(out_dim).to(device))

	def forward(self, feature, edge):
		return edge.mm(feature.mm(self.weight)) + self.bias

class Model(torch.nn.Module):
	def __init__(self, num_features, device):
		super(Model, self).__init__()

		self.norms = torch.nn.ModuleList()
		self.layers = torch.nn.ModuleList()
		self.activation = torch.nn.Tanh()

		# we fix last demension to 1 because our task is a regression
		for in_dim, out_dim in zip(num_features, num_features[1:] + [1]):
			self.norms.append(torch.nn.LayerNorm(in_dim).to(device))
			self.layers.append(Layer(in_dim, out_dim, device))

	def forward(self, feature, edge, mask):
		out = feature * mask
		for norm, layer in zip(self.norms[:-1], self.layers[:-1]):
			out = norm(out) * mask
			out = layer(out, edge) * mask
			out = self.activation(out) * mask
		out = self.norms[-1](out) * mask
		return self.layers[-1](out, edge) * mask
