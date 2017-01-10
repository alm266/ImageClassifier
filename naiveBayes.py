# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
	self.legalLabels = legalLabels
	self.type = "naivebayes"
	self.k = 1 # this is the smoothing parameter, ** use it in your train method **
	self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
	
  def setSmoothing(self, k):
	"""
	This is used by the main method to change the smoothing parameter before training.
	Do not modify this method.
	"""
	self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
	"""
	Outside shell to call your method. Do not modify this method.
	"""	 
	  
	# might be useful in your code later...
	# this is a list of all features in the training set.
	self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
	
	if (self.automaticTuning):
		kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
	else:
		kgrid = [self.k]
		
	self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
	  
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
	"""
	Trains the classifier by collecting counts over the training data, and
	stores the Laplace smoothed estimates so that they can be used to classify.
	Evaluate each value of k in kgrid to choose the smoothing parameter 
	that gives the best accuracy on the held-out validationData.
	
	trainingData and validationData are lists of feature Counters.	The corresponding
	label lists contain the correct label for each datum.
	
	To get the list of all possible features or labels, use self.features and 
	self.legalLabels.
	"""

	self.conditional_prob_table = util.Counter()
	self.prior_distribution_prob_table = util.Counter()
	self.conditional_prob = {}
	self.prior_distribution_prob = {}
	iteration = 0 # use to symbolize which k value we are for storage in dict, since k is a float
	
	#iterate through all k values to obtain best probabilities prior/cond
	for k in kgrid:
		print "K value: ", k, " Iteration: ", iteration
		# helper function to calculate prior dist prob and conditional prob
		self.trainAndTuneHelper(trainingData, trainingLabels, k) 
		# Set weights in tables before next iteration, save all data to choose highest probabilities at the end
		self.conditional_prob_table[iteration] = self.conditional_prob 
		self.prior_distribution_prob_table[iteration] = self.prior_distribution_prob
		iteration += 1 #represents k value since k is a float 
		
	# Assign highest conditional probability and prior distribution probability, pull from dictionaries
	self.conditional_prob = self.conditional_prob_table[max(self.conditional_prob_table, key=self.conditional_prob_table.get)]
	self.prior_distribution_prob = self.prior_distribution_prob_table[max(self.prior_distribution_prob_table, key=self.prior_distribution_prob_table.get)]
  
  
  # Sets up three dimensional dictionary/array for conditional probability values
  def setUpConditionalProb(self, trainingLabels, trainingData):
	conditional_prob = util.Counter()
	for index in range(len(trainingLabels)):
		conditional_prob[index] = util.Counter()
		data = trainingData[index]
		#label = trainingLabels[index]
		for feature, value in data.items():
			conditional_prob[index][feature] = util.Counter()
			conditional_prob[index][feature][value] = 0
	return conditional_prob
	
  def trainAndTuneHelper(self, trainingData, trainingLabels, k):
  
	totalLabels = len(trainingLabels)
	label_occurrence_prob = 1.0/float(totalLabels) #used for each occurrence of a specific label, used to get c(y)/n
	prior_distribution_prob = util.Counter() #holds the prior probability for each legal label
	conditional_prob = self.setUpConditionalProb(trainingLabels, trainingData) #holds the conditional probabiility
	
	# prior distribution probability of legal labels, uniform label probability 1/n where n is the total number of training labels
	for i in range(totalLabels):
		label = trainingLabels[i]
		prior_distribution_prob[label] += label_occurrence_prob
	
	#Begin setup for calculation of P(F=f|Y = y) for every feature value that belongs to 0 or 1
	#Iterate through every image then get there feature dictionary
	#Get the label, key and value as a key to the conditional prob dict, 
	#the value here is to store how many times a specific label at a specific x,y point has a value 0 or 1
	
	for index in range(len(trainingData)):
		#label = trainingLabels[index]
		data = trainingData[index]
		for feature, value in data.items():
			conditional_prob[index][feature][value] += 1
			
	####################### STARTING HERE NOT SURE IF CORRECT ###################################
	
	
	# now we need to add k to all values of c(f,y) 
	# need to get (sum(P(fi|y)))
	sum0 = 0 
	sum1 = 1
	
	for index in range(len(trainingData)):
		#label = trainingLabels[index]
		data = trainingData[index]
		for feature, value in data.items():
			if value == 0:
				sum0 += conditional_prob[index][feature][0]
				conditional_prob[index][feature][0] += k
				sum0 += 1
			else:
				sum1 += conditional_prob[index][feature][1]
				conditional_prob[index][feature][1] += k
	
	for index in range(len(trainingData)):
		#label = trainingLabels[index]
		data = trainingData[index]
		for feature, value in data.items():
			conditional_prob[index][feature][value] /= ((float(sum0)*conditional_prob[index][feature][0])+(conditional_prob[index][feature][1]*float(sum1)))
	
	#at this point in the algorithm we have P(F=f|Y = y) = c(f,y) /sum(feature = 0, 1) * c(f,y)
	
	
	################################## NOT SURE IF THIS PART IS CORRECT ############################################
	
	#update current probabilities for this iteration of k 
	self.conditional_prob = conditional_prob 
	self.prior_distribution_prob = prior_distribution_prob
	
  def classify(self, testData):
	"""
	Classify the data based on the posterior distribution over labels.
	
	You shouldn't modify this method.
	"""
	guesses = []
	self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
	for datum in testData:
	  posterior = self.calculateLogJointProbabilities(datum)
	  guesses.append(posterior.argMax())
	  self.posteriors.append(posterior)
	return guesses
	  
  def calculateLogJointProbabilities(self, datum):
	"""
	Returns the log-joint distribution over legal labels and the datum.
	Each log-probability should be stored in the log-joint counter, e.g.	
	logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
	
	To get the list of all possible features or labels, use self.features and 
	self.legalLabels.
	"""
	logJoint = util.Counter()
	#want to calculate log(P(y)) + log(sum(P(fi|y)))
	#where y is a label
	for label in self.legalLabels:
		logJoint[label] = math.log(self.prior_distribution_prob[label])
		for feature, value in datum.items():
			cp = self.conditional_prob[label][feature][value]
			if cp > 0: #condition check for values < 0 because log(0) is undefined and math domain error occurs
				logJoint[label] += math.log(cp) #summing up
				
	return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
	"""
	Returns the 100 best features for the odds ratio:
			P(feature=1 | label1)/P(feature=1 | label2) 
	
	Note: you may find 'self.features' a useful way to loop through all possible features
	"""
	featuresOdds = []
	return featuresOdds
	

	
	  
