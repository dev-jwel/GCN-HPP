import os
import json
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from loguru import logger
import time, datetime
from math import radians, sin, cos, asin, sqrt, floor

def distance(pos1, pos2):
	long1 = radians(pos1[0])
	lat1 = radians(pos1[1])
	long2 = radians(pos2[0])
	lat2 = radians(pos2[1])

	dlong = long2 - long1
	dlat = lat2 - lat1

	# Haversine formula
	temp = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlong/2)**2

	#radius of the earth is 6371km
	return 6371 * 2 * asin(sqrt(temp))

def gen_edges(longitudes, latitudes, limit, epsilon, self_weight):
	assert len(longitudes) == len(latitudes)
	n = len(longitudes)
	edges = []

	for i in tqdm(range(n), 'calulating edges'):
		for j in range(n):
			i_pos = longitudes[i], latitudes[i]
			j_pos = longitudes[j], latitudes[j]

			d = distance(i_pos, j_pos)

			if d < limit:
				if i == j:
					edges.append((i, j, self_weight/(d+epsilon)))
				else:
					edges.append((i, j, 1/(d+epsilon)))

	return edges

def laplacian(edges, num_node):
	raw = torch.LongTensor(edges)

	indices = raw[:, 0:2].T
	values = raw[:, 2]

	diagonal_indices = torch.arange(num_node)
	diagonal_indices = torch.stack([diagonal_indices, diagonal_indices], dim=1).T

	degree = torch.zeros(num_node)
	for edge in edges:
		degree[edge[0]] += edge[2]

	raw_edge = torch.sparse.FloatTensor(indices, values, (num_node, num_node)).to(torch.float).coalesce()
	degree_matrix = torch.sparse.FloatTensor(diagonal_indices, (degree ** (-0.5)), (num_node, num_node)).to(torch.float).coalesce()

	return torch.sparse.mm(degree_matrix, torch.sparse.mm(raw_edge , degree_matrix)).coalesce()

def gen_prefix(distance_limit, train_rate, val_rate, epsilon, self_weight):
	return '{}-{}-{}-{}-{}'.format(distance_limit, train_rate, val_rate, epsilon, self_weight)

def load_dataset(fname):
	with open(fname, 'r') as f:
		data = json.load(f)

	feature_tensor = torch.Tensor(data['feature_tensor'])
	target = torch.Tensor(data['target'])

	indices = torch.Tensor(data['edge_tensor']['indices']).to(torch.int64)
	values = torch.Tensor(data['edge_tensor']['values'])
	edge_tensor = torch.sparse.FloatTensor(indices, values).coalesce()

	mask = {}
	for k in data['mask']:
		mask[k] = torch.Tensor(data['mask'][k])

	return feature_tensor, edge_tensor, target, mask

def dump_dataset(feature_tensor, edge_tensor, target, mask, fname):
	data = {}
	data['feature_tensor'] = feature_tensor.numpy().tolist()
	data['target'] = target.numpy().tolist()

	data['edge_tensor'] = {}
	data['edge_tensor']['indices'] = edge_tensor.indices().numpy().tolist()
	data['edge_tensor']['values'] = edge_tensor.values().numpy().tolist()

	data['mask'] = {}
	for k in mask:
		data['mask'][k] = mask[k].numpy().tolist()

	with open(fname, 'w') as f:
		json.dump(data, f)

class California():
	def gen_dataset(self, dataframe, distance_limit, train_rate, val_rate, epsilon, self_weight):
		# drop some rows if some column is nan

		for col in dataframe.columns:
			dataframe = dataframe[dataframe[col].notna()]

		# preprocess some features

		dataframe['log_total_rooms'] = np.log(dataframe['total_rooms'])
		dataframe['log_total_bedrooms'] = np.log(dataframe['total_bedrooms'])
		dataframe['log_population'] = np.log(dataframe['population'])
		dataframe['log_households'] = np.log(dataframe['households'])

		for type in dataframe['ocean_proximity'].unique():
			dataframe[type] = (dataframe['ocean_proximity'] == type)
		dataframe = dataframe.drop(columns=['ocean_proximity'])

		# create edges by distance

		longitudes = dataframe.loc[:,'longitude'].tolist()
		latitudes = dataframe.loc[:,'latitude'].tolist()
		edges = gen_edges(longitudes, latitudes, distance_limit, epsilon, self_weight)

		# detach target

		target = torch.Tensor(dataframe['median_house_value'].tolist()).reshape(-1, 1)
		dataframe = dataframe.drop(columns=['median_house_value'])

		# check data size

		num_node, input_dim = dataframe.shape

		# generate graph

		feature_tensor = torch.Tensor(dataframe.values.astype(float))
		edge_tensor = laplacian(edges, num_node)

		# sample train, val, test set

		mask = {}

		mask['train'] = torch.zeros(num_node, 1)
		mask['val'] = torch.zeros(num_node, 1)
		mask['test'] = torch.zeros(num_node, 1)

		train_val = floor(train_rate * num_node)
		val_test = floor((train_rate+val_rate) * num_node)

		indices = torch.randperm(num_node)

		mask['train'][indices[:train_val], 0] = 1
		mask['val'][indices[train_val:val_test], 0] = 1
		mask['test'][indices[val_test:], 0] = 1

		return feature_tensor, edge_tensor, target, mask

	def __init__(self, dir, distance_limit, train_rate, val_rate, epsilon, self_weight, device):
		prefix = gen_prefix(distance_limit, train_rate, val_rate, epsilon, self_weight)
		fname = '{}/california-{}.json'.format(dir, prefix)
		if os.path.exists(fname):
			dataset = load_dataset(fname)
		else:
			dataframe = pd.read_csv('{}/california-housing-prices/housing.csv'.format(dir))
			dataset = self.gen_dataset(dataframe, distance_limit, train_rate, val_rate, epsilon, self_weight)
			dump_dataset(*dataset, fname)
		self.feature_tensor, self.edge_tensor, self.target, self.mask = dataset

		self.feature_tensor = self.feature_tensor.to(device)
		self.edge_tensor = self.edge_tensor.to(device)
		self.target = self.target.to(device)
		self.mask['train'] = self.mask['train'].to(device)
		self.mask['val'] = self.mask['val'].to(device)
		self.mask['test'] = self.mask['test'].to(device)

		logger.debug('num of nodes is {}'.format(self.feature_tensor.shape[0]))
		logger.debug('num of edges is {}'.format(self.edge_tensor.indices().shape[1]))
		logger.debug('size of input dimension is {}'.format(self.feature_tensor.shape[1]))

class Melbourne():
	def gen_dataset(self, dataframe, distance_limit, train_rate, val_rate, epsilon, self_weight):
		# drop NaN for Price, geometric information

		dataframe = dataframe[dataframe['Price'].notna()]
		dataframe = dataframe[dataframe['Lattitude'].notna()]
		dataframe = dataframe[dataframe['Longtitude'].notna()]

		#fill NaN as mean of log for 'BuildingArea'

		cliped = dataframe['BuildingArea'].clip(1) # map zero to one
		dataframe['LogBuildingArea'] = np.log(cliped)
		dataframe['LogBuildingArea'] = dataframe['LogBuildingArea'].fillna(dataframe['LogBuildingArea'].mean())
		dataframe = dataframe.drop(columns=['BuildingArea'])

		#fill NaN as mean for 'BuildingArea'

		dataframe['YearBuilt'] = dataframe['YearBuilt'].fillna(dataframe['YearBuilt'].mean())

		#fill NaN as mean of log for 'BuildingArea'

		cliped = dataframe['Landsize'].clip(1) # map zero to one
		dataframe['LogLandsize'] = np.log(cliped)
		dataframe['LogLandsize'] = dataframe['LogLandsize'].fillna(dataframe['LogLandsize'].mean())
		dataframe = dataframe.drop(columns=['Landsize'])

		# drop other NaNs

		for col in dataframe.columns:
			dataframe = dataframe[dataframe[col].notna()]

		# preprocess some columns

		timestamp = [time.mktime(datetime.datetime.strptime(d, "%d/%m/%Y").timetuple()) for d in dataframe['Date']]
		dataframe['timestamp'] = timestamp
		dataframe = dataframe.drop(columns=['Date'])

		columns_to_encode = ['Rooms', 'Type', 'Method', 'Bedroom2', 'Bathroom', 'Car', 'CouncilArea', 'Regionname']
		for column in columns_to_encode:
			for type in dataframe[column].unique():
				dataframe[column + '_' + str(type)] = (dataframe[column] == type)
		dataframe = dataframe.drop(columns=columns_to_encode)

		columns_to_drop = ['Suburb', 'Address', 'SellerG', 'Postcode']
		dataframe = dataframe.drop(columns=columns_to_drop)

		# create edges by distance

		longitudes = dataframe.loc[:,'Longtitude'].tolist()
		latitudes = dataframe.loc[:,'Lattitude'].tolist()
		edges = gen_edges(longitudes, latitudes, distance_limit, epsilon, self_weight)

		# detach target

		target = torch.Tensor(dataframe['Price'].tolist()).reshape(-1, 1)
		dataframe = dataframe.drop(columns=['Price'])

		# check data size

		num_node, input_dim = dataframe.shape

		# generate graph

		feature_tensor = torch.Tensor(dataframe.values.astype(float))
		edge_tensor = laplacian(edges, num_node)

		# sample train, val, test set

		mask = {}

		mask['train'] = torch.zeros(num_node, 1)
		mask['val'] = torch.zeros(num_node, 1)
		mask['test'] = torch.zeros(num_node, 1)

		train_val = floor(train_rate * num_node)
		val_test = floor((train_rate+val_rate) * num_node)

		indices = torch.randperm(num_node)

		mask['train'][indices[:train_val], 0] = 1
		mask['val'][indices[train_val:val_test], 0] = 1
		mask['test'][indices[val_test:], 0] = 1

		return feature_tensor, edge_tensor, target, mask

	def __init__(self, dir, distance_limit, train_rate, val_rate, epsilon, self_weight, device):
		prefix = gen_prefix(distance_limit, train_rate, val_rate, epsilon, self_weight)
		fname = '{}/melbourne-{}.json'.format(dir, prefix)
		if os.path.exists(fname):
			dataset = load_dataset(fname)
		else:
			dataframe = pd.read_csv('{}/melbourne-housing-market/Melbourne_housing_FULL.csv'.format(dir))
			dataset = self.gen_dataset(dataframe, distance_limit, train_rate, val_rate, epsilon, self_weight)
			dump_dataset(*dataset, fname)
		self.feature_tensor, self.edge_tensor, self.target, self.mask = dataset

		self.feature_tensor = self.feature_tensor.to(device)
		self.edge_tensor = self.edge_tensor.to(device)
		self.target = self.target.to(device)
		self.mask['train'] = self.mask['train'].to(device)
		self.mask['val'] = self.mask['val'].to(device)
		self.mask['test'] = self.mask['test'].to(device)

		logger.debug('num of nodes is {}'.format(self.feature_tensor.shape[0]))
		logger.debug('num of edges is {}'.format(self.edge_tensor.indices().shape[1]))
		logger.debug('size of input dimension is {}'.format(self.feature_tensor.shape[1]))
