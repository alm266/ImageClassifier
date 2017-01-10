# dataClassifier.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# This file contains feature extraction methods and harness 
# code for data classification

import mostFrequent
import naiveBayes
import perceptron
import mira
import samples
import sys
import util
import time
import math

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70

FACE_TRAIN = 451
FACE_TEST = 150
DIGIT_TRAIN = 5000
DIGIT_TEST = 1000


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.Counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...
  
  ##
  """
  features =  basicFeatureExtractorDigit(datum)

  # Test to see if digit contains a loop, 6, 8, 9, create more white
  # Test to see continous white space 1, 2, 3, 5, 7
  
  #checking for continuous white space break up grid into four quadrants, the more white space a 
  #quadrant has the more likely the more continuous white space exists there
  
  totalPoints = 0
  whiteSpace = 0
  wsq1 = 0
  wsq2 = 0
  wsq3 = 0
  wsq4 = 0
  
  #first quadrant
  for x in range(int(0.5*DIGIT_DATUM_WIDTH)):
    for y in range(int(0.5*DIGIT_DATUM_HEIGHT)):
      totalPoints += 1
      if datum.getPixel(x, y) == 0:
        wsq1 += 1
  #second quadrant
  for x in range(int(0.5*DIGIT_DATUM_WIDTH)):
    for y in range(int(0.5*DIGIT_DATUM_HEIGHT), DIGIT_DATUM_HEIGHT):
      totalPoints += 1
      if datum.getPixel(x, y) == 0:
        wsq2 += 1	
  #third quadrant
  for x in range(int(0.5*DIGIT_DATUM_WIDTH), DIGIT_DATUM_WIDTH):
    for y in range(int(0.5*DIGIT_DATUM_HEIGHT), DIGIT_DATUM_HEIGHT):
      totalPoints += 1
      if datum.getPixel(x, y) == 0:
        wsq3 += 1
  #fourth quadrant
  for x in range(int(0.5*DIGIT_DATUM_WIDTH), DIGIT_DATUM_WIDTH):
    for y in range(int(0.5*DIGIT_DATUM_HEIGHT)):
      totalPoints += 1
      if datum.getPixel(x, y) == 0:
        wsq3 += 1
  wsq1 /= float(totalPoints)
  wsq2 /= float(totalPoints)
  wsq3 /= float(totalPoints)
  wsq4 /= float(totalPoints)
  
  temp = [ wsq1 > 0.50, wsq2 > 0.50, wsq3 > 0.50, wsq4 > 0.50 ]
  features[('ws','q1')] = temp[0]
  features[('ws','q2')] = temp[1]
  features[('ws','q3')] = temp[2]
  features[('ws','q4')] = temp[3]
  
  #check against which numbers would produce this much white space in those quadrants
  #check for loops
  
  return features


def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  features =  basicFeatureExtractorFace(datum)
  
  # Test if left side of potential face has the same amount of points as right side
  # Test if inside of potential face has less points than outside, outside should have more points 
  # for hair, ears, mouth, oval shape etc... vs. nose, parts of eyes
  
  symmetric = False
  leftSide = 0
  total = 0
  for x in range(int(0.5*FACE_DATUM_WIDTH)):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        total += 1
        leftSide += 1
  
  for x in range(int(0.5*FACE_DATUM_WIDTH), FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        total += 1
  
  percent = float(leftSide)/float(total)
  if (percent) > .40 and (percent) < .60:
    symmetric = True

  features[('symmetric', 'left')] = symmetric
  
  inside = 0
  outside = False
  
  for x in range(int(0.3*FACE_DATUM_WIDTH), int(0.7*FACE_DATUM_WIDTH)):
    for y in range(int(0.3*FACE_DATUM_HEIGHT), int(0.7*FACE_DATUM_HEIGHT)):
      if datum.getPixel(x, y) > 0:
        leftSide += 1
  
  percent = float(inside)/float(total)
  if (percent) < .50 and (percent) > .20:
    outside = True
  
  features[('outside', 'morepts')] = outside
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print "==================================="
          print "Mistake on example %d" % i 
          print "Predicted %d; truth is %d" % (prediction, truth)
          print "Image: "
          print rawTestData[i]
          break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print "new features:", pix
            continue
      print image  

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['mostFrequent', 'nb', 'naiveBayes', 'perceptron', 'mira', 'minicontest'], default='mostFrequent')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-w', '--weights', help=default('Whether to print weights'), default=False, action="store_true")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-s', '--test', help=default("Amount of test data to use"), default=TEST_SET_SIZE, type="int")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print "Doing classification"
  print "--------------------"
  print "data:\t\t" + options.data
  print "classifier:\t\t" + options.classifier
  if not options.classifier == 'minicontest':
    print "using enhanced features?:\t" + str(options.features)
  else:
    print "using minicontest feature extractor"
  #print "training set size:\t" + str(options.training)
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
    if (options.classifier == 'minicontest'):
      featureFunction = contestFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace      
  else:
    print "Unknown dataset", options.data
    print USAGE_STRING
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = range(10)
  else:
    legalLabels = range(2)
    
  if options.training <= 0:
    print "Training set size should be a positive integer (you provided: %d)" % options.training
    print USAGE_STRING
    sys.exit(2)
    
  if options.smoothing <= 0:
    print "Please provide a positive number for smoothing (you provided: %f)" % options.smoothing
    print USAGE_STRING
    sys.exit(2)
    
  if options.odds:
    if options.label1 not in legalLabels or options.label2 not in legalLabels:
      print "Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2)
      print USAGE_STRING
      sys.exit(2)

  if(options.classifier == "mostFrequent"):
    classifier = mostFrequent.MostFrequentClassifier(legalLabels)
  elif(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print "using automatic tuning for naivebayes"
        classifier.automaticTuning = True
    else:
        print "using smoothing parameter k=%f for naivebayes" %  options.smoothing
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  elif(options.classifier == "mira"):
    classifier = mira.MiraClassifier(legalLabels, options.iterations)
    if (options.autotune):
        print "using automatic tuning for MIRA"
        classifier.automaticTuning = True
    else:
        print "using default C=0.001 for MIRA"
  elif(options.classifier == 'minicontest'):
    import minicontest
    classifier = minicontest.contestClassifier(legalLabels)
  else:
    print "Unknown classifier:", options.classifier
    print USAGE_STRING
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage
  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
      
  # Load data  
  numTraining = options.training
  numTest = options.test
  
  # ADDED CODE
  print "\n\n\n"
  
  percentage = 10
  ErrorEstimates = []
  AccuracyEstimates = []
  trainSize = 0
  testSize = 0
  totalTime = 0
  while percentage <= 100:
    print "TRAINING WITH ", percentage, " percent of TRAINING DATA"
    if (options.data=="faces"):
        trainSize = int(float(percentage)/float(100) * FACE_TRAIN)
        testSize = FACE_TEST
        print "TRAINING SET SIZE: ", trainSize, "TOTAL TRAINING SIZE: ", FACE_TRAIN
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", trainSize,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", trainSize)
        rawTestData = samples.loadDataFile("facedata/facedatatest", testSize,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", testSize)
        rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    else:
        trainSize = int(float(percentage)/float(100) * DIGIT_TRAIN)
        testSize = DIGIT_TEST
        print "TRAINING SET SIZE: ", trainSize, "TOTAL TRAINING SIZE: ", DIGIT_TRAIN
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", trainSize,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", trainSize)
        rawTestData = samples.loadDataFile("digitdata/testimages", testSize,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", testSize)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    percentage += 10
    print "Extracting features..."
    trainingData = map(featureFunction, rawTrainingData)
    testData = map(featureFunction, rawTestData)
    validationData = map(featureFunction, rawValidationData)
    print "Training..."
    start = time.time()
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    end = time.time()
    print "TOTAL TIME ELAPSED FOR ALGORITHM TRAINING: ", end-start, " secs"
    totalTime += (end-start)
    if options.data == "faces":
        print "Testing set total size:", FACE_TEST
    else:
        print "Testing set total size:", DIGIT_TEST
    print "Testing..."
    guesses = classifier.classify(testData)
    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
    errorPercentage = 100.0 - ((100.0 * correct / len(testLabels)))
    ErrorEstimates.append(errorPercentage)
    AccuracyEstimates.append((100.0 * correct / len(testLabels)))
    print "Error Percentage: ", errorPercentage
    print "\n\n\n"
  #print "END OF WHILE LOOP"
  print "TOTAL TIME FOR ALL ITERATIONS 10, 20, ... , 100 percent of DATA to be TRAINED: ", totalTime, "secs\n"
  print "Accuracy List for 10%, 20%, ... , 100%:"
  print AccuracyEstimates
  mean = float(sum(AccuracyEstimates))/float(len(AccuracyEstimates))
  temp = [ float((num-mean)**2) for num in AccuracyEstimates ]
  mean = float(sum(temp))/float(len(AccuracyEstimates))
  sd = math.sqrt(mean)
  print "MEAN FOR ACCURACY: ", mean
  print "STANDARD DEVIATION FOR ACCURACY: ", sd, "\n"
  print "Error Estimate List for 10%, 20%, ... , 100%:"
  print ErrorEstimates, "\n"
  mean2 = float(sum(ErrorEstimates))/float(len(ErrorEstimates))
  temp2 = [ float((i-mean)**2) for i in ErrorEstimates ]
  mean2 = float(sum(temp2))/float(len(ErrorEstimates))
  sd2 = math.sqrt(mean)
  print "MEAN FOR ERROR ESTIMATES: ", mean2
  print "STANDARD DEVIATION FOR ERRORS: ", sd2, "\n"
  
  
  #END OF ADDED CODE
    

  """if(options.data=="faces"):
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTest)
    rawTestData = samples.loadDataFile("facedata/facedatatest", numTest,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numTest)
  else:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", numTest)
    rawTestData = samples.loadDataFile("digitdata/testimages", numTest,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)
    
  
  # Extract features
  print "Extracting features..."
  trainingData = map(featureFunction, rawTrainingData)
  validationData = map(featureFunction, rawValidationData)
  testData = map(featureFunction, rawTestData)
  
  # Conduct training and testing
  print "Training..."
  classifier.train(trainingData, trainingLabels, validationData, validationLabels)
  
  print "Validating..."
  guesses = classifier.classify(validationData)
  correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
  print str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels))
  
  print "Testing..."
  guesses = classifier.classify(testData)
  correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
  print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))
  analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
  
  # do odds ratio computation if specified at command line
  if((options.odds) & (options.classifier == "naiveBayes" or (options.classifier == "nb")) ):
    label1, label2 = options.label1, options.label2
    features_odds = classifier.findHighOddsFeatures(label1,label2)
    if(options.classifier == "naiveBayes" or options.classifier == "nb"):
      string3 = "=== Features with highest odd ratio of label %d over label %d ===" % (label1, label2)
    else:
      string3 = "=== Features for which weight(label %d)-weight(label %d) is biggest ===" % (label1, label2)    
      
    print string3
    printImage(features_odds)

  if((options.weights) & (options.classifier == "perceptron")):
    for l in classifier.legalLabels:
      features_weights = classifier.findHighWeightFeatures(l)
      print ("=== Features with high weight for label %d ==="%l)
      printImage(features_weights)"""

if __name__ == '__main__':
  # Read input
  args, options = readCommand( sys.argv[1:] ) 
  # Run classifier
  runClassifier(args, options)
