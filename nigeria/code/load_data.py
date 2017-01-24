'''Functions and tools for loading and preprocessing the data.
The functions take the data from the .csv file and manipulate it as a pandas dataframe and as
a numpy array.'''

from __future__ import print_function

import numpy as np
import pandas as pd

def ReadHeader(fname='_data/headers.csv'):

	'''Read the header file, output is a dictionary which maps column ID to content description.'''

	header = {}

	## Loop through the file
	skip = set([0,1])
	header_file = open(fname,'r')
	for i,line in enumerate(header_file):

		## Skip lines in set skip
		if i in skip:
			continue

		## Otherwise, parse the line
		l = line.split(',')

		## And store it in the dictionary
		header[l[0].strip().lower()] = l[1]
	header_file.close()

	## two entries are missing from the header.csv file.
	## Here, I put in filler for those two to avoid issues later.
	header['var1'] = 'var1'
	header['caseid'] = 'case id'
	return header

def ReadData(fname='_data/data.csv'):

	'''Function to read the data. Output is a pandas dataframe'''

	## Start by constructing the headers for the pandas data frame
	## Open the file, read the first line, strip the \n and \r, and parse.
	data = open(fname,'r')
	columns = data.readline().rstrip().lower().split(',')
	data.close()

	## Now use pandas to handle the rest
	df = pd.read_csv(fname,skiprows=0)
	df.columns = columns
	return df

def CleanData(df):

	'''Perform some cleaning operations to make the data easier to work with.'''

	## Compress the "i don't know" and "missing" categories
	df['h8'].replace([8.,9.],np.nan,inplace=True)
	df['h0'].replace([8.,9.],np.nan,inplace=True)
	df['h9'].replace([8.,9.],np.nan,inplace=True)


	## Make all the known ones equal
	df['h8'].replace([1.,2.,3.],1.,inplace=True)
	df['h0'].replace([1.,2.,3.],1.,inplace=True)
	df['h9'].replace([1.,2.,3.],1.,inplace=True)

def LoadData(dataset='_data/data.csv',header_file='_data/headers.csv',resample=False):

	'''Wrapper for the functions above.

	The resample option uses the statistical weights in column v005 and samples (with replacement) a 
	fraction of the data provided by the user.'''

	header = ReadHeader(header_file)
	df = ReadData(dataset)

	## Resample using the weights column
	if resample:
		df = df.sample(frac=resample,replace=True,weights=df['v005'])

	## Simplify the vaccine columns of interest.
	CleanData(df)

	return header, df



if __name__ == "__main__":

	df = ReadData()

	#print df.isnull().sum()['hw1']
	hist = df['hw1'].value_counts()









