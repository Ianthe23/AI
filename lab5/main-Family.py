import csv
import os
import matplotlib.pyplot as plt
import numpy as np  
from sklearn import linear_model

# ------------------------------------BY Family - UNIVARIATE LINEAR REGRESSION------------------------------------

# load data and consider a single feature (Family) and the output to be estimated (happiness)

def loadDataGBP(fileName, inputVariabName, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable = dataNames.index(inputVariabName)
    inputs = [float(data[i][selectedVariable]) for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    
    return inputs, outputs

crtDir =  os.getcwd()
filePath = os.path.join(crtDir, 'data', 'v3_world-happiness-report-2017.csv')

inputs, outputs = loadDataGBP(filePath, 'Family', 'Happiness.Score')
print('in:  ', inputs[:5])
print('out: ', outputs[:5])


# see how the data looks (plot the histograms associated to input data - Family feature - and output data - happiness)

# def plotDataHistogramGBP(x, variableName):
#     n, bins, patches = plt.hist(x, 10)
#     plt.title('Histogram of ' + variableName)
#     plt.show()

# plotDataHistogramGBP(inputs, 'Family')
# plotDataHistogramGBP(outputs, 'Happiness score')


# check the liniarity (to check that a linear relationship exists between the dependent variable (y = happiness) and the independent variable (x = capita).)

# plt.plot(inputs, outputs, 'ro') 
# plt.xlabel('Family')
# plt.ylabel('happiness')
# plt.title('Family vs. happiness')
# plt.show()


# Split the Data Into Training and Test Subsets
# In this step we will split our dataset into training and testing subsets (in proportion 80/20%).

# Training data set is used for learning the linear model. Testing dataset is used for validating of the model. 
# All data from testing dataset will be new to model and we may check how accurate are model predictions.

np.random.seed(5)
indexes = [i for i in range(len(inputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
validationSample = [i for i in indexes  if not i in trainSample]

trainInputs = [inputs[i] for i in trainSample]
trainOutputs = [outputs[i] for i in trainSample]

validationInputs = [inputs[i] for i in validationSample]
validationOutputs = [outputs[i] for i in validationSample]

# plt.plot(trainInputs, trainOutputs, 'ro', label = 'training data')   #train data are plotted by red and circle sign
# plt.plot(validationInputs, validationOutputs, 'g^', label = 'validation data')     #test data are plotted by green and a triangle sign
# plt.title('train and validation data')
# plt.xlabel('Family')
# plt.ylabel('happiness')
# plt.legend()
# plt.show()



#learning step: init and train a linear regression model y = f(x) = w0 + w1 * x
#Prediction step: used the trained model to estimate the output for a new input

# #using sklearn 
# #training data preparation (the sklearn linear model requires as input training data as noSamples x noFeatures array; in the current case, the input must be a matrix of len(trainInputs) lineas and one columns (a single feature is used in this problem))
# xx = [[el] for el in trainInputs]

# #model initialisation
# regressor = linear_model.LinearRegression()
# #training the model by using the training inputs and known training outputs
# regressor.fit(xx, trainOutputs)
# #save the model parameters
# w0, w1 = regressor.intercept_, regressor.coef_[0]
# print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x')

# using developed code
from utils.functions import MyLinearUnivariateRegression

# model initialisation
regressor = MyLinearUnivariateRegression()
# training the model by using the training inputs and known training outputs
regressor.fit(trainInputs, trainOutputs)
# save the model parameters
w0, w1 = regressor.intercept_, regressor.coef_
print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x')



# plot the learnt model
# prepare some synthetic data (inputs are random, while the outputs are computed by the learnt model)
noOfPoints = 1000
xref = []
val = min(trainInputs)
step = (max(trainInputs) - min(trainInputs)) / noOfPoints
for i in range(1, noOfPoints):
    xref.append(val)
    val += step
yref = [w0 + w1 * el for el in xref] 

plt.plot(trainInputs, trainOutputs, 'ro', label = 'training data')  #train data are plotted by red and circle sign
plt.plot(xref, yref, 'b-', label = 'learnt model')                  #model is plotted by a blue line
plt.title('train data and the learnt model')
plt.xlabel('Family')
plt.ylabel('happiness')
plt.legend()
plt.show()



# use the trained model to predict new inputs

# makes predictions for test data (manual)
# computedTestOutputs = [w0 + w1 * el for el in testInputs]

# makes predictions for test data (by tool)
computedValidationOutputs = regressor.predict([[x] for x in validationInputs])

# plot the computed outputs (see how far they are from the real outputs)
# plt.plot(validationInputs, computedValidationOutputs, 'yo', label = 'computed test data')  #computed test data are plotted yellow red and circle sign
# plt.plot(validationInputs, validationOutputs, 'g^', label = 'real test data')  #real test data are plotted by green triangles
# plt.title('computed validation and real validation data')
# plt.xlabel('Family')
# plt.ylabel('happiness')
# plt.legend()
# plt.show()



# compute the differences between the predictions and real outputs
# "manual" computation
error = 0.0
for t1, t2 in zip(computedValidationOutputs, validationOutputs):
    error += (t1 - t2) ** 2
error = error / len(validationOutputs)
print('prediction error (manual): ', error)

# by using sklearn 
from sklearn.metrics import mean_squared_error

error = mean_squared_error(validationOutputs, computedValidationOutputs)
print('prediction error (tool):  ', error)








