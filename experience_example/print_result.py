from loguru import logger
from fire import Fire
import json

def main():
	datasets = ['california', 'melbourne']
	models = ['mlp', 'gcn']
	dims_for_dataset = {'california': [2, 4, 8, 16, 32], 'melbourne': [16, 32, 64, 128]}

	result = {}

	with open('result.json', 'r') as f:
		result = json.load(f)

	for dataset in datasets:
		dims = dims_for_dataset[dataset]
		for model in models:
			for dim in dims:
				single_result = result[dataset][model][str(dim)]
				score = single_result['metric']['R_square']
				print('dataset:{}, model:{}, dim: {}, score:{}'.format(dataset, model, dim, score))

if __name__ == '__main__':
	Fire(main)
