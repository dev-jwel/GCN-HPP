from fire import Fire
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
import time, datetime

def analyze_california(dir):
	df = pd.read_csv("{}/california-housing-prices/housing.csv".format(dir))

	logger.info("Dataset has been loaded. Its size is {}.".format(df.shape[0]))

	logger.info("First, let's check nan data on this dataset.")

	print(df.isna().sum())

	logger.info("There are few nan data on 'total_bedrooms'. I will just drop it.")

	df = df[df['total_bedrooms'].notna()]

	logger.info("Size of dropped dataset is {}.".format(df.shape[0]))
	logger.info("Second, let's check some columns which value has log scale.")
	logger.info("Please close the plot window after you checked it.")

	fig = plt.figure()

	ax1 = fig.add_subplot(2,2,1)
	ax1.hist(df['total_rooms'])
	ax1.set_title('total_rooms')
	ax2 = fig.add_subplot(2,2,2)
	ax2.hist(df['total_bedrooms'])
	ax2.set_title('total_bedrooms')
	ax3 = fig.add_subplot(2,2,3)
	ax3.hist(df['population'])
	ax3.set_title('population')
	ax4 = fig.add_subplot(2,2,4)
	ax4.hist(df['households'])
	ax4.set_title('households')

	plt.show()

	logger.info("To normalize it let's check log of previous data.")

	fig = plt.figure()

	ax1 = fig.add_subplot(2,2,1)
	ax1.hist(np.log(df['total_rooms']))
	ax1.set_title('total_rooms')
	ax2 = fig.add_subplot(2,2,2)
	ax2.hist(np.log(df['total_bedrooms']))
	ax2.set_title('total_bedrooms')
	ax3 = fig.add_subplot(2,2,3)
	ax3.hist(np.log(df['population']))
	ax3.set_title('population')
	ax4 = fig.add_subplot(2,2,4)
	ax4.hist(np.log(df['households']))
	ax4.set_title('households')

	plt.show()

	logger.info("Much better!")

	logger.info("Let's check 'ocean_proximity' column before genarete its one hot encoding.")

	print(df['ocean_proximity'].unique())

	logger.info("Finally preprocessed dataset is below.")

	df['log_total_rooms'] = np.log(df['total_rooms'])
	df['log_total_bedrooms'] = np.log(df['total_bedrooms'])
	df['log_population'] = np.log(df['population'])
	df['log_households'] = np.log(df['households'])
	df = df.drop(columns=['total_rooms', 'total_bedrooms', 'population', 'households'])

	for type in df['ocean_proximity'].unique():
		df[type] = (df['ocean_proximity'] == type)
	df = df.drop(columns=['ocean_proximity'])

	logger.info("print dataset")

	logger.info("* columns")
	print(df.columns)
	logger.info("* head()")
	print(df.head())
	logger.info("* shape")
	print(df.shape)

def analyze_melbourne(dir):
	df = pd.read_csv("{}/melbourne-housing-market/Melbourne_housing_FULL.csv".format(dir))

	logger.info('Before we use this dataset, drop NaN price because we want to predict it.')

	df = df[df['Price'].notna()]

	logger.info("And also we generates graph using geometric information so let's drop NaN for this.")

	df = df[df['Lattitude'].notna()]
	df = df[df['Longtitude'].notna()]

	logger.info("Dataset has been loaded. Its size is {}.".format(df.shape[0]))

	# handle NaN datas

	logger.info("First, let's check nan data on this dataset.")

	print(df.isna().sum())

	logger.info("Oh, there are too many NaNs!")
	logger.info("Almost half of 'BuildingArea' is NaN!")
	logger.info("Before handle this, let's check this not NaN values.")

	building_area = df[df['BuildingArea'].notna()]['BuildingArea']
	plt.hist(building_area)
	plt.show()

	logger.info("Even this value has log scale!")
	logger.info("let's check log of this value again.")

	filtered_building_area = building_area.where(building_area>0).dropna()
	plt.hist(np.log(filtered_building_area))
	plt.show()

	logger.info("Quite nice distribution. Let's fill NaN to mean of log scale.")

	cliped = df['BuildingArea'].clip(1) # map zero to one
	df['LogBuildingArea'] = np.log(cliped)
	df['LogBuildingArea'] = df['LogBuildingArea'].fillna(df['LogBuildingArea'].mean())
	df = df.drop(columns=['BuildingArea'])

	logger.info("Let's check distribution again.")

	plt.hist(df['LogBuildingArea'])
	plt.show()

	logger.info("Very sharp distribution... But it is better than NaN.")
	logger.info("Let's check NaN again.")

	print(df.isna().sum())

	logger.info("We still have a problem on 'YearBuilt' and 'Landsize'.")
	logger.info("Let's check 'YearBuilt'.")

	year_built = df[df['YearBuilt'].notna()]['YearBuilt']
	plt.hist(year_built)
	plt.show()

	logger.info("I think there is outlier which is 1196. Let's drop it.")

	filtered_year_built = year_built.where(year_built>1196).dropna()
	plt.hist(filtered_year_built)
	plt.show()

	logger.info("Smooth distribution. Let's map NaN to mean.")

	df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].mean())

	logger.info("Last one to map NaN is 'Landsize'. I will drop other NaNs.")

	land_size = df[df['Landsize'].notna()]['Landsize']
	plt.hist(land_size)
	plt.show()

	logger.info("This is log scale. Preprocess it same as 'BuildingArea'.")

	cliped = df['Landsize'].clip(1) # map zero to one
	df['LogLandsize'] = np.log(cliped)
	df['LogLandsize'] = df['LogLandsize'].fillna(df['LogLandsize'].mean())
	df = df.drop(columns=['Landsize'])

	logger.info("Its distribution is:")

	plt.hist(df['LogLandsize'])
	plt.show()

	logger.info("We mapped major NaN columns. Drop others.")

	for col in df.columns:
		df = df[df[col].notna()]

	logger.info("Now its size is {}.".format(df.shape[0]))

	# preprocess some columns

	logger.info("We have 'Date' column. Let's change this into unix timestamp.")

	timestamp = [time.mktime(datetime.datetime.strptime(d, "%d/%m/%Y").timetuple()) for d in df['Date']]
	df['timestamp'] = timestamp
	df = df.drop(columns=['Date'])

	logger.info("We have some columns that has small set of values.")
	logger.info("Let's check these before perform one hot encoding.")

	columns_to_encode = ['Rooms', 'Type', 'Method', 'Bedroom2', 'Bathroom', 'Car', 'CouncilArea', 'Regionname']
	for column in columns_to_encode:
		print(df[column].unique())

	logger.info("One hot encoding for these columns:")

	for column in columns_to_encode:
		for type in df[column].unique():
			df[column + '_' + str(type)] = (df[column] == type)
	df = df.drop(columns=columns_to_encode)

	logger.info("Finally, drop meaningless columns.")

	columns_to_drop = ['Suburb', 'Address', 'SellerG', 'Postcode']
	df = df.drop(columns=columns_to_drop)

	logger.info("print dataset")

	logger.info("* columns")
	print(df.columns)
	logger.info("* head()")
	print(df.head())
	logger.info("* shape")
	print(df.shape)

def main(dir='data', dataset='california'):
	if dataset == 'california':
		analyze_california(dir)
	elif dataset == 'melbourne':
		analyze_melbourne(dir)
	else:
		raise NotImplementedError('no such dataset {}'.format(dataset))

if __name__ == '__main__':
	Fire(main)
