'''Answer the basic data questions.'''

from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data
import statsmodels.formula.api as smf

## Colors for plots [in RGB]
red = (145./255.,3./255.,3./255.)
blue = (0.,84./255.,156./255.)

def Histogram(df,column_name='hw1',save=False,show=True,verbose=False):

	'''Compute and plot a 1d histogram for column name.'''

	## Get a numpy array of the value counts (ignores NaNs)
	hist = df[column_name].value_counts().sort_index(axis=0)
	if verbose:
		print(hist)

	## Plot if specified
	if save or show:

		values = hist.index
		
		## Type convert if catagorical
		if values.dtype.name.startswith('int') or values.dtype.name.startswith('float'):
			x = np.array(values.tolist())
		else:
			type_dict = {v:i for i,v in enumerate(values)}
			print('Data in '+column_name+' is categorical! Here are the catagories:')
			print(type_dict)
			x = np.array([type_dict[v] for v in values])

		## Get the counts
		counts = hist.tolist()
		plt.figure()
		plt.step(x,counts,where='mid',lw=2.,c=red)
		plt.xlabel('Values')
		plt.ylabel('Counts')
		plt.title('Histrogram for '+column_name)
		plt.xlim((min(x)-1.,max(x)+1.))
		plt.ylim((min(counts)*0.5,max(counts)*1.05))
		if save:
			plt.savefig('histogram_'+column_name+'.pdf')
		if show:
			plt.show()

	return hist

def ScatterPlot(df,x_col,y_col='h0',save=False,show=True,jitter_y=True,jitter_x=False):

	## Retrieve the y data
	y = df[y_col].as_matrix()

	## Get the x_data. Check for data type. If the data is catagorical,
	## assign each catagory a number and plot accordingly.
	if df[x_col].dtype.name.startswith('int') or df[x_col].dtype.name.startswith('float'):
		x = df[x_col].as_matrix()
	else:
		types = df[x_col].value_counts().sort_index(axis=0).index.tolist()
		type_dict = {t:i for i,t in enumerate(types)}
		print('Data in '+x_col+' is categorical! Here are the catagories:')
		print(type_dict)
		x = df[x_col].tolist()
		x = np.array([type_dict[t] for t in x])

	## Adding slight noise to the data helps with visualization.
	if jitter_y:
		y = y + np.random.normal(scale=max(y)*0.05,size=y.shape)
	if jitter_x:	
		try:	
			x = x + np.random.normal(scale=max(x)*0.05,size=x.shape)
		except ValueError:
			x = x + np.random.normal(scale=0.05,size=x.shape)

	plt.figure()
	plt.scatter(x,y,c='0.25',alpha=0.5)
	plt.xlabel(x_col)
	plt.ylabel(y_col)
	plt.title('Scatter plot of '+y_col+' vs. '+x_col)
	if save:
		plt.savefig(x_col+"_vs_"+y_col+".pdf")
	if show:
		plt.show()

def PredictivePower(df,header,y_col='h0',verbose=True):

	'''Function to do single variable logistic regression on a dataset. Independent variable is
	y_col. Output is a sorted list of the most predictive variables as measured by pseudo R^2.'''


	## Storage for the results
	test_results = []

	## Loop through the columns excluding 
	## ones we want to avoid.
	## For each we try a single variable logistic regression
	## and store the results.
	exclude = {'caseid','h0','h8'}
	for name in df.columns:
		print('Testing column '+name)
		if name in exclude:
			print('Skipping this one!')
			continue
		try:
			formula = y_col+" ~ "+name
			model = smf.logit(formula, data=df)
			results = model.fit(full_output=False,disp=False)

		except (ValueError, TypeError, np.linalg.linalg.LinAlgError):
			print('Failed to test! Description: '+header[name])
			continue

		test_results.append([results.prsquared,name,results.llr_pvalue])

	## Sort the results by by R^2
	test_results.sort(key=lambda x: x[0],reverse=True)

	## Print the top 10 results
	if verbose:
		print('\n')
		print('Top predictors are:')
		print('[R^2, column name, p-value, description]')
		for regression in test_results[:10]:
			print(regression+[header[regression[1]]])
	
	return test_results

def CorrelationMatrix(df,v1,v2,verbose=True):

	'''Function to print the correlation matrix for two columns in the dataframe. Assigns numerical
	values to categorical data if needed.'''

	if df[v1].dtype.name.startswith('int') or df[v1].dtype.name.startswith('float'):
		x = df[v1]
	else:
		types = df[v1].value_counts().sort_index(axis=0).index.tolist()
		type_dict = {t:i for i,t in enumerate(types)}
		x = df[v1].replace(type_dict,inplace=False)

	if df[v2].dtype.name.startswith('int') or df[v2].dtype.name.startswith('float'):
		y = df[v2]
	else:
		types = df[v2].value_counts().sort_index(axis=0).index.tolist()
		type_dict = {t:i for i,t in enumerate(types)}
		y = df[v2].replace(type_dict,inplace=False)	

	new_df = pd.concat([x,y],axis=1)
	corr = new_df.corr()
	if verbose:
		print(new_df.corr())
	return corr

if __name__ == "__main__":

	## Get the data using the functions in
	## load_data.py
	header, df = load_data.LoadData()

	## Make some plots
	hist = Histogram(df,column_name='hw1',verbose=False,show=True,save=False)
	#print(df['hw1'].isnull().sum())
	#print(df[df['b5']=='No']['hw1'].isnull().sum()/float(df['hw1'].isnull().sum()))



	## Fraction alive
	number_of_children = len(df)
	number_alive = len(df[df['b5']=='Yes'])
	fraction_alive = float(number_alive)/number_of_children
	print('There are',number_alive,'children alive out of',number_of_children,
		'total. That is',100.*fraction_alive,'percent.')

	## Scatter plots
	#ScatterPlot(df[df['sstate']=='BAUCHI'],'sstate',jitter_y=True,jitter_x=True,save=False)


	## randomize and split into training, test, and validation sets
	## 60 percent for training, 20 for testing, 20 for validating
	#train, validation, test = np.split(df.sample(frac=1.), [int(.6*len(df)), int(.8*len(df))])
	
	t = PredictivePower(df,header,y_col='h8')


	












