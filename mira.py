# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
PRINT = True

class MiraClassifier:
  """
  Mira classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
	self.legalLabels = legalLabels
	self.type = "mira"
	self.automaticTuning = False 
	self.C = 0.001
	self.legalLabels = legalLabels
	self.max_iterations = max_iterations
	self.initializeWeightsToZero()

  def initializeWeightsToZero(self):
	"Resets the weights of each label to zero vectors" 
	self.weights = {}
	for label in self.legalLabels:
	  self.weights[label] = util.Counter() # this is the data-structure you should use
  
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
	"Outside shell to call your method. Do not modify this method."	 
	  
	self.features = trainingData[0].keys() # this could be useful for your code later...
	
	if (self.automaticTuning):
		Cgrid = [0.002, 0.004, 0.008]
	else:
		Cgrid = [self.C]
		
	return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
	"""
	This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
	then store the weights that give the best accuracy on the validationData.
	
	Use the provided self.weights[label] data structure so that 
	the classify method works correctly. Also, recall that a
	datum is a counter from features to values for those features
	representing a vector of values.
	"""
	
	#iterate through all c values in the Cgrid
	#use train and tune helper method to find highest score label and compare to true label for every datum, similar to perceptron
	#if true label != highest scored label then will adjust weights with tau
	#after returning the weights for this iteration of c weights will be updated in the weights dictionary
	#select the best weights for self.weights to be tested from the weight dictionary, use iteration to keep track of best c value 
	weights = util.Counter()
	iteration = 1
	for c in Cgrid:
		self.trainAndTuneHelper(trainingData, trainingLabels, c)
		weights[iteration] = self.weights
		iteration += 1
	#Store the best weights we received from the best value of c/iteration 
	self.weights = weights[max(weights, key=weights.get)]
	
  def trainAndTuneHelper(self, trainingData, trainingLabels, c):
  
	#similar to the perceptron algorithm, get scores for all labels, find highest score, check against true label
	#if highest scored label != true label adjust weights accordingly with Tau
	
	scores = util.Counter()
	for iteration in range(self.max_iterations):
		print "C Value: ", c, "Iteration: ", iteration
		for index in range(len(trainingData)):
			data = trainingData[index]
			y = trainingLabels[index]
			for label in self.legalLabels:
				labelWeight = self.weights[label]
				scores[label] = data * labelWeight
			yPrime = max(scores, key=scores.get)
			# here call helper method if y != yPrime to adjust weights 
			if y != yPrime:
				self.adjustWeightsTau(trainingData, trainingLabels, scores, y, yPrime, data, c)
	
  def adjustWeightsTau(self, trainingData, trainingLabels, scores, y, yPrime, data, c):
	
	# calculate Tau
	numerator = float(((self.weights[yPrime] - self.weights[y]) * data)) + 1
	denominator = 2 * float((data*data))
	tauTemp = numerator/denominator
	
	# cap the max with c value
	tau = min(c, tauTemp)
	
	# multiply all values of f by tau in the data
	# adjust weights after obtaining values
	weightsToAddSub = util.Counter()
	# in order to make weightsToAddSub addable or subtractable 
	for key in data.keys():
		weightsToAddSub[key] = data[key]*tau
	self.weights[y] += weightsToAddSub
	self.weights[yPrime] -= weightsToAddSub
	
  def classify(self, data ):
	"""
	Classifies each datum as the label that most closely matches the prototype vector
	for that label.	 See the project description for details.
	
	Recall that a datum is a util.counter... 
	"""
	guesses = []
	for datum in data:
	  vectors = util.Counter()
	  for l in self.legalLabels:
		vectors[l] = self.weights[l] * datum
	  guesses.append(vectors.argMax())
	return guesses

  
  def findHighOddsFeatures(self, label1, label2):
	"""
	Returns a list of the 100 features with the greatest difference in feature values
					 w_label1 - w_label2

	"""
	featuresOdds = []
	return featuresOdds

