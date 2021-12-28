import sys
from fire import Fire
from loguru import logger
import data

def main(
	model='gcn',
	dir='data',
	dataset='california',
	device='cpu',
	epochs=100000,
	lr=0.01,
	reg=10,
	hidden_dim=32,
	distance_limit=2.5,
	train_rate=0.85,
	val_rate=0.05,
	early_stopping=2000,
	epsilon=1e-3,
	self_weight=100
):

	if dataset == 'california':
		dataset = data.California(dir, distance_limit, train_rate, val_rate, epsilon, self_weight, device)
	elif dataset == 'melbourne':
		dataset = data.Melbourne(dir, distance_limit, train_rate, val_rate, epsilon, self_weight, device)
	else:
		logger.error("no such dataset {}".format(dataset))
		return None

	if model == 'gcn':
		from gcn.model import Model
		from gcn.eval import Evaluator
		from gcn.train import Trainer
		from gcn.loss import MaskedMSELoss

		input_dim = dataset.feature_tensor.shape[1]
		model = Model([input_dim, hidden_dim], device)
		evaluator = Evaluator()
		lossfn = MaskedMSELoss()
		trainer = Trainer(model, dataset, lr, reg, evaluator, lossfn)
	elif model == 'mlp':
		from mlp.model import Model
		from mlp.eval import Evaluator
		from mlp.train import Trainer
		from mlp.loss import MaskedMSELoss

		input_dim = dataset.feature_tensor.shape[1]
		model = Model([input_dim, hidden_dim], device)
		evaluator = Evaluator()
		lossfn = MaskedMSELoss()
		trainer = Trainer(model, dataset, lr, reg, evaluator, lossfn)
	else:
		logger.error('no such model {}'.format(model))
		return None

	trained_model, history = trainer.train(epochs, early_stopping)
	metrics = evaluator(trained_model, dataset, 'test')

	logger.info('metrics are: {}'.format(metrics))
	return metrics, history

def main_wraper(**kwargs):
	main(**kwargs)

if __name__ == "__main__":
	Fire(main_wraper)
