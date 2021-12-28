import torch
from loguru import logger

class Evaluator():
	def __call__(self, model, dataset, mode):
		mask = dataset.mask[mode]
		N = mask.sum()
		target = dataset.target
		output = model(dataset.feature_tensor, mask)

		SSE = ((mask * (target - output)) ** 2).sum()
		mean = (mask * target).sum() / N
		var = ((mask * (target - mean)) ** 2).sum()

		R_square = 1 - SSE / var

		return {'R_square': R_square.item()}
