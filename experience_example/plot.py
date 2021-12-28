from loguru import logger
from fire import Fire
import json
from matplotlib import pyplot as plt

def main():
	datasets = ['california', 'melbourne']
	models = ['mlp', 'gcn']
	dims_for_dataset = {'california': [2, 4, 8, 16, 32], 'melbourne': [16, 32, 64, 128]}

	result = {}

	with open('result.json', 'r') as f:
		result = json.load(f)

	for dataset in datasets:
		dims = dims_for_dataset[dataset]
		for score in ['loss', 'R_square']:
			for model in models:

				prefix = '{}-{}-{}'.format(score, dataset, model)
				fig, axs = plt.subplots(1, len(dims))
				fig.set_size_inches(8*len(dims), 6)

				for i, dim in enumerate(dims):
					history = result[dataset][model][str(dim)]['history']
					scores = [h[score] for h in history]
					axs[i].plot(range(len(scores)), scores)
					if score == 'loss':
						axs[i].set_yscale('log')
					else:
						axs[i].set_yscale('linear')
				fig.savefig('{}.png'.format(prefix))
				fig.clf()

if __name__ == '__main__':
	Fire(main)
