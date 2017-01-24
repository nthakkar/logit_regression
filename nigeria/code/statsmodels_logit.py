'''StatsModels implementation of the logistic regression on the Nigerian family survey data.'''

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import load_data
import statsmodels.formula.api as smf
from patsy import dmatrix
import six.moves.cPickle as pickle


##################################
## Regression Classes
##################################

## Super class
class StatsModelLogit(object):

	def __init__(self,train_df,predictors,y_col):

		## Save the predictor and y_col lists
		self.predictors = predictors
		self.y_col = y_col

		## Generate and store the Patsy formula
		self.formula = y_col[0]+' ~ '+predictors[0]+''.join([' + '+s for s in predictors[1:]])

		## Create and fit model
		self.model = smf.logit(self.formula,data=train_df)
		self.results = self.model.fit()

	def predict(self,test_df,rounding=True):

		## Compute probabilites
		test_df_subset = test_df[self.predictors].dropna()
		predicted_probabilites = self.results.predict(test_df_subset)

		## Round to find prediction
		if rounding:
			return np.array(np.round(predicted_probabilites))
		else:
			return np.array(predicted_probabilites)

	def error_test(self,test_df):

		## Compute probablities and actual values
		test_subset = test_df[self.predictors+self.y_col].dropna()
		actual = np.reshape(test_subset[self.y_col].as_matrix(),(len(test_subset),))
		predicted = self.predict(test_subset)

		## Compute errors
		errors = sum(np.abs(actual-predicted))
		percent_error = (100.*errors)/len(test_subset)
		print("Percent error on the test set =",percent_error)

	def __str__(self):

		## Use StatsModels built in summary
		print(self.results.summary())

		## Return an empty string to avoid the python error
		return ''

## Child class to handle some issues
## with pickling statsmodels and pandas dataframes
class PickleLogit(StatsModelLogit):

	def __init__(self,fname,predictors,y_col):

		## Save the predictor and y_col lists
		self.predictors = predictors
		self.y_col = y_col

		## Get the results from the model
		self.results = pickle.load(open(fname))

		## Generate and store the Patsy formula
		self.formula = y_col[0]+' ~ '+predictors[0]+''.join([' + '+s for s in predictors[1:]])

	def predict(self,test_df,rounding=True):

		## Get the appropriate data
		test_df_subset = test_df[self.predictors].dropna()

		## Compute the probabilities
		## This is slightly different because smf.model.fit() adds attributes
		## to the dataframe that need to be added explicitly here.
		x = dmatrix(self.formula[self.formula.find("~")+2:], data=test_df_subset)  
		predicted_probabilites = self.results.predict(x,transform=False)

		## Round to find prediction
		if rounding:
			return np.array(np.round(predicted_probabilites))
		else:
			return np.array(predicted_probabilites)


##################################
## Helper Functions
##################################
def SaveModel(classifier,name='_model'):

	## Make the directory for the model
	os.system('rm -r '+name)
	os.system('mkdir '+name)

	## Pickle the necessary objects
	## This has to be done this way since pickling the classifier directly
	## causes errors due to conflicts with pandas. 
	pickle.dump(classifier.predictors,open(name+'/predictors.pkl','wb'))
	pickle.dump(classifier.y_col,open(name+'/y_col.pkl','wb'))
	classifier.results.save(name+'/'+classifier.y_col[0]+name+'.pkl')

def LoadModel(name='_model'):

	## Load the predictor and y_col lists
	predictors = pickle.load(open(name+'/predictors.pkl'))
	y_col = pickle.load(open(name+'/y_col.pkl'))

	## Load the model using the Pickle child class of 
	## StatsModelsLogit
	classifier = PickleLogit(name+'/'+y_col[0]+name+'.pkl',predictors,y_col)
	return classifier

##################################
## Wrapper for other data
##################################
def Predict(dataset = '_data/data.csv',header_file='_data/headers.csv',model_dir='_model'):

	'''Wrapper function to load a pickled model and the dataset and perform an error test.'''

	## Get the data
	print('...loading the data')
	header, df = load_data.LoadData(dataset,header_file)

	## Unpickle the model
	print('...retrieving the model')
	classifier = LoadModel(model_dir)
	print(classifier)

	## Perform error test
	classifier.error_test(df)



if __name__ == "__main__":

	header, df = load_data.LoadData()

	## Split into testing and training data.
	## 80 percent is training, 20 percent is testing.
	## This is done with random resampling.
	train, test = np.split(df.sample(frac=1.), [int(.8*len(df))])


	## Train a new model:
	predictors = ['sstate','v106','v190','hw1','h9']
	classifier = StatsModelLogit(train,predictors,['h0'])
	print(classifier)
	classifier.error_test(test)
	SaveModel(classifier)


	## Load and test an old model
	#classifier = LoadModel(name='_model')
	#print(classifier)
	#classifier.error_test(test)
	#Predict()


















