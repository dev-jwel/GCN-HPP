import math
import torch
import random
import numpy as np
from tqdm.auto import tqdm
from loguru import logger
from copy import deepcopy

class Trainer:
	def __init__(self, model, dataset, lr, reg, evaluator, lossfn):
		self.model = model
		self.dataset = dataset
		self.lr = lr
		self.reg = reg
		self.evaluator = evaluator
		self.lossfn = lossfn

	def train(self, epochs, early_stopping):
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)

		best_model = deepcopy(self.model)
		best_val_epoch = 0
		best_val_score = -float('Inf')
		history = []

		for epoch in tqdm(range(epochs), desc='epoch'):
			optimizer.zero_grad()
			self.model.train()
			output = self.model(self.dataset.feature_tensor, self.dataset.mask['train'])
			loss = self.lossfn(output, self.dataset.target, self.dataset.mask['train'])
			loss.backward()
			optimizer.step()

			self.model.eval()
			with torch.no_grad():
				val_measures = self.evaluator(self.model, self.dataset, 'val')

			val_measures['loss'] = loss.item()
			val_score = val_measures['R_square']
			history.append(val_measures)

			if val_score > best_val_score:
				best_val_score = val_score
				best_val_epoch = epoch
				best_model = deepcopy(self.model)

			if epoch - best_val_epoch > early_stopping:
				break

		return best_model, history
