from __future__ import print_function

import six.moves.cPickle as pickle
import os
import timeit
import numpy as np
import pandas as pd

import theano
import theano.tensor as T
from load_data import *

##################################
## Regression Class
##################################
class LogisticRegression(object):

	'''input is the data - needs to be a theano.tensor.TensorType. Use PortData below to 
	handle the type conversion.'''

	def __init__(self,input,n_in,n_out):

		## Allocate shared variables for the parameters
		self.W = theano.shared(value=np.zeros((n_in,n_out),dtype=theano.config.floatX),
							   name='W', borrow=True)

		self.b = theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX),
							   name='b',borrow=True)


		## Compile theano function for the probabilities
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		## And then the prediction given the probabilities. This is just done by choosing
		## the most probable.
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		## Store the input and the parameters
		self.params = [self.W,self.b]
		self.input = input

	def negative_log_likelihood(self,y):

		## We use the average log_likelihood instead of the total. This minimizes 
		## dependence on the data batch size.
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

	def errors(self,y):

		'''Perform an error test for known results y, again expected as a theano tensor type.'''

		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as y_pred!')

		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()


##################################
## Pandas to Theano Functions
##################################
def PortData(dataset='_data/data.csv',header_file='_data/headers.csv',
			predictors=['sstate','v106','v190','hw1','h9'],y=['h0']):

	'''Function to interface pandas with theano. Basically just converts the loaded data
	from load_data.py into a set of theano tensors that can be manipulated with the logistic regression
	class above, copied to a GPU for fast computation, etc.

	predictors is the subset of the data we're using based on the analysis in basic_data_analysis.py'''

	## Get the dataset as a pandas dataframe
	## Cut out the portion of the data we want,
	## including dropping nan's.
	header, df = LoadData(dataset,header_file)
	df = df[predictors+y].dropna()

	## Restructure the df to have dummy variables for each 
	## catagorical variable value. This is done automatically
	## in the statsmodels implementation, but here we have to do it
	## explicitly.
	dummy_predictors = []

	## Loop through each of the predictors
	for predictor in predictors:
		name = df[predictor].dtype.name
		type_check = (name.startswith('int')) or (name.startswith('float'))

		## If the type isn't catagorical, then there's no need to change anything.
		if type_check:
			dummy_predictors.append(predictor)
			continue

		## But if it is, we expand our data frame to have a 1 or 0 column for each 
		## catagory.
		else:
			## Get the different catagories in alphabetical order
			catagories = df[predictor].value_counts().sort_index(axis=0).index.tolist()
			for catagory in catagories:
				## We convert from bool to float for ease later.
				df[catagory] = 1.*(df[predictor]==catagory) 
				dummy_predictors.append(catagory)

	dummy_df = df[dummy_predictors+y]

	## Split the dataset into training (60%), testing (20%), 
	## and validation (20%). This is done with random resampling.
	train_xy, validate_xy, test_xy = np.split(dummy_df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

	def shared_dataset(data_xy, predictors, y, borrow=True):
		
		''' 
		NB: This function is more or less copied from the theano implementation of logistic regression on
		the MNIST data. The only changes have been made to adapt the function to handle pandas df's as
		input.

		Function that loads the dataset into shared variables

		The reason we store our dataset in shared variables is to allow
		Theano to copy it into the GPU memory (when code is run on GPU).
		Since copying data into the GPU is slow, copying a minibatch everytime
		is needed (the default behaviour if the data is not in a shared
		variable) would lead to a large decrease in performance.
		'''

		## Split the dataset and convert to np arrays.
		data_x = data_xy[predictors].as_matrix()
		data_y = np.reshape(data_xy[y].as_matrix(),(len(data_xy),))


		shared_x = theano.shared(np.asarray(data_x,
								 dtype=theano.config.floatX),
								 borrow=borrow)
		shared_y = theano.shared(np.asarray(data_y,
								 dtype=theano.config.floatX),
								 borrow=borrow)
		## When storing data on the GPU it has to be stored as floats
		## therefore we will store the labels as ``floatX`` as well
		## (``shared_y`` does exactly that). But during our computations
		## we need them as ints (we use labels as index, and if they are
		## floats it doesn't make sense) therefore instead of returning
		## ``shared_y`` we will have to cast it to int. This little hack
		## lets ous get around this issue
		return shared_x, T.cast(shared_y, 'int32')

	train_set_x, train_set_y = shared_dataset(train_xy,dummy_predictors,y,borrow=True)
	valid_set_x, valid_set_y = shared_dataset(validate_xy,dummy_predictors,y,borrow=True)
	test_set_x, test_set_y = shared_dataset(test_xy,dummy_predictors,y,borrow=True)

	## Encapsulate and return
	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
	return rval


##################################
## Stochastic gradient descent
##################################
def sgd_optimization(learning_rate=0.13, n_epochs=5000, dataset='_data/data.csv',batch_size=1000):

	## Retrieve the data as theano tensors.
	datasets = PortData(dataset,y=['h0'])

	## Unravel the data.
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	## Determine the dimensionality of the dataset
	n_in = train_set_x.get_value(borrow=True).shape[1]

	# Compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


	print('...building the model')
	## Allocate symbolic minibatch variables
	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	## construct the logistic regression class
	## n_out is 2 since we have preprocessed the data to be either
	## yes or no (as opposed to the original variety of catagories, see
	## the header.csv file for details.)
	classifier = LogisticRegression(input=x, n_in=n_in, n_out=2)

	## Compile a minimization function
	cost = classifier.negative_log_likelihood(y)

	## Compile a theano function for mistakes made
	## on each minibatch
	test_model = theano.function(inputs=[index], outputs=classifier.errors(y),
				givens = {x: test_set_x[index*batch_size:(index+1)*batch_size],
						  y: test_set_y[index*batch_size:(index+1)*batch_size]})


	validate_model = theano.function(inputs=[index], outputs=classifier.errors(y),
				givens = {x: valid_set_x[index*batch_size:(index+1)*batch_size],
						  y: valid_set_y[index*batch_size:(index+1)*batch_size]})

	## Compute the gradient of cost.
	## This is done with symbolic automatic differentiation,
	## and is part of why implementation in theano is worth the trouble.
	g_W = T.grad(cost=cost, wrt=classifier.W)
	g_b = T.grad(cost=cost, wrt=classifier.b)

	## Encapsulate updates
	updates = [(classifier.W, classifier.W - learning_rate*g_W),
			   (classifier.b, classifier.b - learning_rate*g_b)]

	## training function
	train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
					givens = {x: train_set_x[index*batch_size:(index+1)*batch_size],
							  y: train_set_y[index*batch_size:(index+1)*batch_size]})

	print('...training the model')

	## early stopping parameters
	patience = 5000
	patience_increase = 2
	improvement_threshold = 0.995
	validation_frequency = min(n_train_batches, patience // 2)

	best_validation_loss = np.inf
	test_score = 0.
	start_time = timeit.default_timer()

	done_looping = False
	epoch = 0
	while (epoch < n_epochs) and (not done_looping):

		epoch = epoch + 1

		for minibatch_index in range(n_train_batches):

			minibatch_avg_cost = train_model(minibatch_index)
			
			## Look at the validation set?
			iter = (epoch - 1)*n_train_batches + minibatch_index
			if (iter+1)%validation_frequency == 0:

				validation_losses = [validate_model(i) for i in range(n_valid_batches)]
				this_validation_loss = np.mean(validation_losses)

				print('epoch %i, minibatch %i/%i, validation error %f %%' %
					(epoch,minibatch_index+1,n_train_batches,this_validation_loss*100.))

				## if we got the best so far
				if this_validation_loss < best_validation_loss:
					
					## improve patience
					if this_validation_loss < best_validation_loss*improvement_threshold:
						patience = max(patience, iter*patience_increase)

					## Reset the best loss
					best_validation_loss = this_validation_loss

					## How's it going on the test set?
					test_losses = [test_model(i) for i in range(n_test_batches)]
					test_score = np.mean(test_losses)

					print('epoch %i, minibatch %i/%i, test error %f %%' %
						(epoch,minibatch_index+1,n_train_batches,test_score*100.))

					# pickle the best model
					with open('theano_model.pkl','wb') as f:
						pickle.dump(classifier, f)

			if patience <= iter:
				done_looping = True
				break

	end_time = timeit.default_timer()
	print(('Complete! Best validation score of %f %%, with test performace %f %%')%
		(best_validation_loss*100.,test_score*100.))

	print('The code run for %d epochs, with %f epochs/sec' % 
			(epoch, 1. * epoch / (end_time - start_time)))

##################################
## Prediction wrapper
##################################
def predict(dataset='_data/data.csv',header_file = '_data/headers.csv',model_pkl='theano_model.pkl'):

	'''Function to unpickle the model, load in a dataset and header file, and test the model.'''

	## load the model
	print('...unpickling the model')
	classifier = pickle.load(open(model_pkl))

	## compile a predictor funciton
	predict_model = theano.function(inputs=[classifier.input], outputs=classifier.y_pred)

	# We can test it on some examples from test data
	print('...loading the dataset')
	datasets = PortData(dataset,header_file)
	test_set_x, test_set_y = datasets[2]
	test_set_x = test_set_x.get_value()

	predicted_values = predict_model(test_set_x[:10])
	print("Predicted values for the first 10 examples in test set:")
	print(predicted_values)
	print("Actual values for the first 10 examples in the test set:")
	print(test_set_y[:10].eval())

if __name__=="__main__":
	#sgd_optimization()
	predict()






















