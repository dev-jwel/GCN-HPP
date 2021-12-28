import torch
from loguru import logger

class MaskedMSELoss(torch.nn.Module):
	def forward(self, output, target, mask):
		N = mask.sum()
		loss = ( (mask * (target - output)) ** 2).sum() / N
		return loss
