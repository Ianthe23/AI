import warnings; warnings.simplefilter('ignore')
import csv
import matplotlib.pyplot as plt 
import numpy as np 
import os
from mpl_toolkits import mplot3d
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

# Function to plot 3D data for visualization
def plot3Ddata(x1Train, x2Train, yTrain, x1Model=None, x2Model=None, yModel=None, 
              x1Test=None, x2Test=None, yTest=None, title=None):
    ax = plt.axes(projection='3d')
    
    # Only add items to the legend if they're actually plotted
    legend_elements = []
    
    if x1Train and len(x1Train) > 0:
        train_plot = ax.scatter(x1Train, x2Train, yTrain, c='r', marker='o')
        legend_elements.append((train_plot, 'train data'))
    
    if x1Model is not None and len(x1Model) > 0:
        model_plot = ax.scatter(x1Model, x2Model, yModel, c='b', marker='_')
        legend_elements.append((model_plot, 'learnt model'))
    
    if x1Test and len(x1Test) > 0:
        test_plot = ax.scatter(x1Test, x2Test, yTest, c='g', marker='^')
        legend_elements.append((test_plot, 'test data'))
    
    # Only create legend if we have elements to show
    if legend_elements:
        ax.legend([item[0] for item in legend_elements], 
                 [item[1] for item in legend_elements])
    
    plt.title(title)
    ax.set_xlabel("capita")
    ax.set_ylabel("freedom")
    ax.set_zlabel("happiness")
    plt.show()

# Load data with multiple inputs
def loadDataMoreInputs(fileName, inputVariabNames, outputVariabName):
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
    selectedVariable1 = dataNames.index(inputVariabNames[0])
    selectedVariable2 = dataNames.index(inputVariabNames[1])
    inputs = [[float(data[i][selectedVariable1]), float(data[i][selectedVariable2])] for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    
    return inputs, outputs

# Helper function to plot data histograms
def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()

# Main program for happiness prediction based on GDP and Freedom
crtDir = os.getcwd()
filePath = os.path.join(crtDir, 'data', '2017.csv')

# Load GDP, Freedom data and happiness scores
inputs, outputs = loadDataMoreInputs(filePath, ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')

# Extract features
feature1 = [ex[0] for ex in inputs]  # GDP per capita
feature2 = [ex[1] for ex in inputs]  # Freedom

# Plot the data histograms
# plotDataHistogram(feature1, 'GDP per capita')
# plotDataHistogram(feature2, 'Freedom')
# plotDataHistogram(outputs, 'Happiness score')

# Check linearity in 3D
# plot3Ddata(feature1, feature2, outputs, None, None, None, None, None, None, 'GDP vs Freedom vs Happiness')

# Split data into training and test sets (80/20)
np.random.seed(5)
indexes = [i for i in range(len(inputs))]
trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
testSample = [i for i in indexes if i not in trainSample]

trainInputs = [inputs[i] for i in trainSample]
trainOutputs = [outputs[i] for i in trainSample]
testInputs = [inputs[i] for i in testSample]
testOutputs = [outputs[i] for i in testSample]

# Apply custom normalization
normalizedTrainInputs, normalizedTestInputs = my_normalization(trainInputs, testInputs)
normalizedTrainOutputs, normalizedTestOutputs = my_normalization(trainOutputs, testOutputs)

# Extract normalized features for visualization
feature1train = [ex[0] for ex in normalizedTrainInputs]  # Normalized GDP
feature2train = [ex[1] for ex in normalizedTrainInputs]  # Normalized Freedom
feature1test = [ex[0] for ex in normalizedTestInputs]    # Normalized GDP (test)
feature2test = [ex[1] for ex in normalizedTestInputs]    # Normalized Freedom (test)

# Plot normalized training and test data
# plot3Ddata(feature1train, feature2train, normalizedTrainOutputs, 
#           None, None, None, 
#           feature1test, feature2test, normalizedTestOutputs, 
#           "Train and test data (after normalization)")

# Option 1: Using sklearn with batch gradient descent
from sklearn.linear_model import SGDRegressor

# Configure SGDRegressor to use Batch Gradient Descent
skl_regressor = SGDRegressor(learning_rate='constant', eta0=0.01, max_iter=1000, 
                            tol=1e-3, random_state=42, 
                            fit_intercept=True, average=False)
skl_regressor.partial_fit(normalizedTrainInputs, normalizedTrainOutputs)
w0_skl, w1_skl, w2_skl = skl_regressor.intercept_[0], skl_regressor.coef_[0], skl_regressor.coef_[1]
print('\nSKLearn model: f(x) = ', w0_skl, ' + ', w1_skl, ' * x1 + ', w2_skl, ' * x2')

# Option 2: Using custom Batch Gradient Descent implementation
batch_regressor = MyBatchGDRegression(learning_rate=0.01, iterations=1000, batch_size=64)
batch_regressor.partial_fit(normalizedTrainInputs, normalizedTrainOutputs)
w0_custom, w1_custom, w2_custom = batch_regressor.intercept_, batch_regressor.coef_[0], batch_regressor.coef_[1]
print('Custom model: f(x) = ', w0_custom, ' + ', w1_custom, ' * x1 + ', w2_custom, ' * x2')

# Numerical representation of the regression model (surface)
noOfPoints = 50
xref1 = []
val = min(feature1train)
step1 = (max(feature1train) - min(feature1train)) / noOfPoints
for _ in range(1, noOfPoints):
    for _ in range(1, noOfPoints):
        xref1.append(val)
    val += step1

xref2 = []
val = min(feature2train)
step2 = (max(feature2train) - min(feature2train)) / noOfPoints
for _ in range(1, noOfPoints):
    aux = val
    for _ in range(1, noOfPoints):
        xref2.append(aux)
        aux += step2

# Calculate model predictions for the surface
yref = [w0_custom + w1_custom * el1 + w2_custom * el2 for el1, el2 in zip(xref1, xref2)]
yref_skl = [w0_skl + w1_skl * el1 + w2_skl * el2 for el1, el2 in zip(xref1, xref2)]

# Plot the training data with the custom model surface
plot3Ddata(feature1train, feature2train, normalizedTrainOutputs, 
          xref1, xref2, yref, 
          None, None, None, 
          'Train data and the custom model surface')

# Plot the training data with the sklearn model surface
plot3Ddata(feature1train, feature2train, normalizedTrainOutputs, 
          xref1, xref2, yref_skl, 
          None, None, None, 
          'Train data and the sklearn model surface')


# Make predictions on test data
computedTestOutputs = batch_regressor.predict(normalizedTestInputs)

# Plot predictions vs actual test data
# plot3Ddata(None, None, None,
#           feature1test, feature2test, computedTestOutputs,
#           feature1test, feature2test, normalizedTestOutputs,
#           'Predictions vs real test data')

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