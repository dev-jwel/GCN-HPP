from main import main as model_main
from loguru import logger
from fire import Fire
import json

def main(**kwargs):
	datasets = ['california', 'melbourne']
	models = ['mlp', 'gcn']
	dims_for_dataset = {'california': [2, 4, 8, 16, 32], 'melbourne': [16, 32, 64, 128]}
	epochs = 100000
	early_stopping = 2000
	distance_limit_for_dataset = {'california': 3, 'melbourne': 1.5}
	lr = 0.01
	reg = 10
	self_weight_for_datset = {'california': 2500, 'melbourne': 25}

	result = {}

	for dataset in datasets:
		result[dataset] = {}
		dims = dims_for_dataset[dataset]
		for model in models:
			result[dataset][model] = {}
			for dim in dims:
				result[dataset][model][dim] = {}

	with open('result.json', 'w') as f:
		json.dump(result, f)


	for dataset in datasets:
		dims = dims_for_dataset[dataset]
		distance_limit = distance_limit_for_dataset[dataset]
		self_weight = self_weight_for_datset[dataset]

		for model in models:
			for dim in dims:

				msg = 'dataset:{}, model:{}, dim:{}'.format(
					dataset, model, dim
				)

				logger.info(msg)
				with open('log.txt', 'a') as f:
					f.write(msg + '\n')

				metric, history = model_main(
					dataset=dataset,
					model=model,
					hidden_dim=dim,
					epochs=epochs,
					early_stopping=early_stopping,
					distance_limit=distance_limit,
					lr=lr,
					reg=reg,
					self_weight=self_weight,
					**kwargs
				)

				with open('log.txt', 'a') as f:
					f.write('metric: {}'.format(metric) + '\n')

				result[dataset][model][dim] = {'metric':metric, 'history':history}
				with open('result.json', 'w') as f:
					json.dump(result, f)

				print()

if __name__ == '__main__':
	Fire(main)
