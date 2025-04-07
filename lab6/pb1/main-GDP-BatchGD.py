import warnings; warnings.simplefilter('ignore')
import csv
import matplotlib.pyplot as plt 
import numpy as np 
import os
from Batch_GD.batch_GD import MyBatchGDRegression

# Custom normalization function for regression problems
def my_normalization(trainData, testData):
    # Step 1: Identify normalization parameters (mean, stdDev) for training set
    if not isinstance(trainData[0], list):
        # For univariate data
        mean = sum(trainData) / len(trainData)
        stdDev = (sum([(x - mean) ** 2 for x in trainData]) / len(trainData)) ** 0.5
        
        # Step 2: Normalize training data
        normalizedTrainData = [(x - mean) / stdDev for x in trainData]
        
        # Step 3: Normalize test data using same parameters
        normalizedTestData = [(x - mean) / stdDev for x in testData]
    else:
        # For multivariate data
        # Calculate mean and stdDev for each feature
        means = [sum(x[i] for x in trainData) / len(trainData) for i in range(len(trainData[0]))]
        stdDevs = [
            (sum([(x[i] - means[i]) ** 2 for x in trainData]) / len(trainData)) ** 0.5 
            for i in range(len(trainData[0]))
        ]
        
        # Normalize training data
        normalizedTrainData = [
            [(x[i] - means[i]) / stdDevs[i] for i in range(len(x))]
            for x in trainData
        ]
        
        # Normalize test data using same parameters
        normalizedTestData = [
            [(x[i] - means[i]) / stdDevs[i] for i in range(len(x))]
            for x in testData
        ]
    
    return normalizedTrainData, normalizedTestData

# Load data from CSV
def loadData(fileName, inputVariabName, outputVariabName):
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

# Helper function to plot data histograms
def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()

# Main program
crtDir = os.getcwd()
filePath = os.path.join(crtDir, 'data', '2017.csv')

# Load GDP data and happiness scores
inputs, outputs = loadData(filePath, 'Economy..GDP.per.Capita.', 'Happiness.Score')
print('Input GDP samples: ', inputs[:5])
print('Output happiness samples: ', outputs[:5])

# # Plot histograms
# plotDataHistogram(inputs, "GDP per capita")
# plotDataHistogram(outputs, "Happiness score")

# Check linearity between GDP and happiness
# plt.plot(inputs, outputs, 'ro')
# plt.xlabel('GDP per capita')
# plt.ylabel('Happiness score')
# plt.title('GDP per capita vs. Happiness score')
# plt.show()

# Split data into training and test sets (80/20)
np.random.seed(5)
indexes = [i for i in range(len(inputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
testSample = [i for i in indexes if i not in trainSample]

trainInputs = [inputs[i] for i in trainSample]
trainOutputs = [outputs[i] for i in trainSample]
testInputs = [inputs[i] for i in testSample]
testOutputs = [outputs[i] for i in testSample]

# Plot train and test data
# plt.plot(trainInputs, trainOutputs, 'ro', label='training data')
# plt.plot(testInputs, testOutputs, 'g^', label='testing data')
# plt.title('Train and test data')
# plt.xlabel('GDP per capita')
# plt.ylabel('Happiness score')
# plt.legend()
# plt.show()

# Normalize data using custom function
normalizedTrainInputs, normalizedTestInputs = my_normalization(trainInputs, testInputs)
normalizedTrainOutputs, normalizedTestOutputs = my_normalization(trainOutputs, testOutputs)

# Prepare input format for regression model
xx = [[el] for el in normalizedTrainInputs]

# Option 1: Using sklearn with batch gradient descent
from sklearn.linear_model import SGDRegressor

# Configure SGDRegressor to use Batch Gradient Descent
# Setting batch_size to len(xx) makes it full batch gradient descent
skl_regressor = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000, 
                            tol=1e-3, random_state=42, 
                            fit_intercept=True, average=False)
skl_regressor.partial_fit(xx, normalizedTrainOutputs)
w0_skl, w1_skl = skl_regressor.intercept_[0], skl_regressor.coef_[0]
print('\nSKLearn model: f(x) = ', w0_skl, ' + ', w1_skl, ' * x')

# Option 2: Using custom Batch Gradient Descent implementation
batch_regressor = MyBatchGDRegression(learning_rate=0.01, iterations=1000, batch_size=64)
batch_regressor.partial_fit(xx, normalizedTrainOutputs)
w0_custom, w1_custom = batch_regressor.intercept_, batch_regressor.coef_[0]
print('Custom model: f(x) = ', w0_custom, ' + ', w1_custom, ' * x')

# Plot the learned model (using custom implementation)
noOfPoints = 1000
xref = []
val = min(normalizedTrainInputs)
step = (max(normalizedTrainInputs) - min(normalizedTrainInputs)) / noOfPoints
for i in range(1, noOfPoints):
    xref.append(val)
    val += step
    
yref = [w0_custom + w1_custom * el for el in xref]

# Plot both models (sklearn and custom)
yref_skl = [w0_skl + w1_skl * el for el in xref]

plt.figure(figsize=(10, 6))
plt.plot(normalizedTrainInputs, normalizedTrainOutputs, 'ro', label='training data')
plt.plot(xref, yref, 'b-', label='custom model')
plt.plot(xref, yref_skl, 'g--', label='sklearn model')
plt.title('Train data and both learned models (normalized)')
plt.xlabel('GDP per capita (normalized)')
plt.ylabel('Happiness score (normalized)')
plt.legend()
plt.show()

# Make predictions on test data
computedTestOutputs = batch_regressor.predict([[x] for x in normalizedTestInputs])

# Plot predictions vs actual test data
# plt.plot(normalizedTestInputs, computedTestOutputs, 'yo', label='predicted data')
# plt.plot(normalizedTestInputs, normalizedTestOutputs, 'g^', label='actual test data')
# plt.title('Predicted vs actual test data (normalized)')
# plt.xlabel('GDP per capita (normalized)')
# plt.ylabel('Happiness score (normalized)')
# plt.legend()
# plt.show()

# Calculate prediction error
error = 0.0
for t1, t2 in zip(computedTestOutputs, normalizedTestOutputs):
    error += (t1 - t2) ** 2
error = error / len(normalizedTestOutputs)
print('\nPrediction error (manual): ', error)

# Using sklearn's mean_squared_error
from sklearn.metrics import mean_squared_error
error = mean_squared_error(normalizedTestOutputs, computedTestOutputs)
print('Prediction error (sklearn): ', error) 