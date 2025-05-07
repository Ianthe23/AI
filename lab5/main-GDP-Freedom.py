import csv
import os
import matplotlib.pyplot as plt
import numpy as np  
from sklearn import linear_model

# ------------------------------------BY GBP AND FREEDOM - BIVARIATE LINEAR REGRESSION------------------------------------

# load data and consider two features (Economy..GDP.per.Capita. and Freedom) and the output to be estimated (Happiness.Score)
def loadDataGBPFreedom(fileName, inputVariable1, inputVariable2, outputVariable):
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
        selectedVariable1 = dataNames.index(inputVariable1)
        selectedVariable2 = dataNames.index(inputVariable2)
        selectedOutput = dataNames.index(outputVariable)
        
        # Filter out rows with missing or invalid values
        filtered_inputs = []
        filtered_outputs = []
        for i in range(len(data)):
            try:
                # Check if any value is empty
                if data[i][selectedVariable1] == '' or data[i][selectedVariable2] == '':
                    continue
                    
                # Try to convert to float to catch any invalid values
                gdp = float(data[i][selectedVariable1])
                freedom = float(data[i][selectedVariable2])
                happiness = float(data[i][selectedOutput])
                
                # Only include non-zero values
                if gdp > 0 and freedom > 0:
                    filtered_inputs.append([gdp, freedom])
                    filtered_outputs.append(happiness)
            except ValueError:
                # Skip row if conversion to float fails
                continue
        
        print(f"Total rows in dataset: {len(data)}")
        print(f"Rows after filtering: {len(filtered_inputs)}")
        print(f"Rows removed due to missing/invalid data: {len(data) - len(filtered_inputs)}")
        
        return filtered_inputs, filtered_outputs
    
crtDir =  os.getcwd()
filePath = os.path.join(crtDir, 'data', 'v3_world-happiness-report-2017.csv')
    
inputs, outputs = loadDataGBPFreedom(filePath, 'Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score')
print('in:  ', inputs[:5])
print('out: ', outputs[:5])

# see how the data looks (plot the histograms associated to input data - GDP feature - and output data - happiness)

# def plotDataHistogramGBPFreedom(x, variableName):
#     # Convert to list if it's a numpy array
#     if isinstance(x, np.ndarray):
#         x = x.tolist()
#     n, bins, patches = plt.hist(x, 10)
#     plt.title('Histogram of ' + variableName)
#     plt.show()

# inputs = np.array(inputs)
# plotDataHistogramGBPFreedom(inputs[:, 0], 'GDP capita')
# plotDataHistogramGBPFreedom(inputs[:, 1], 'Freedom')
# plotDataHistogramGBPFreedom(outputs, 'Happiness score')


# # check the liniarity (to check that a linear relationship exists between the dependent variable (y = happiness) and the independent variables (x = capita and freedom).)
# plt.figure(figsize=(12, 5))

# # First subplot for GDP vs Happiness
# plt.subplot(1, 2, 1)
# plt.plot(inputs[:, 0], outputs, 'ro') 
# plt.xlabel('GDP capita')
# plt.ylabel('happiness')
# plt.title('GDP capita vs. happiness')

# # Second subplot for Freedom vs Happiness
# plt.subplot(1, 2, 2)
# plt.plot(inputs[:, 1], outputs, 'ro') 
# plt.xlabel('Freedom')
# plt.ylabel('happiness')
# plt.title('Freedom vs. happiness')

# plt.tight_layout()  # Adjust layout to prevent overlap
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

# # Create a figure with two subplots
# plt.figure(figsize=(12, 5))

# # First subplot for GDP vs Happiness
# plt.subplot(1, 2, 1)
# plt.plot([x[0] for x in trainInputs], trainOutputs, 'ro', label='training data')
# plt.plot([x[0] for x in validationInputs], validationOutputs, 'g^', label='validation data')
# plt.xlabel('GDP capita')
# plt.ylabel('happiness')
# plt.title('GDP capita vs. happiness')
# plt.legend()

# # Second subplot for Freedom vs Happiness
# plt.subplot(1, 2, 2)
# plt.plot([x[1] for x in trainInputs], trainOutputs, 'ro', label='training data')
# plt.plot([x[1] for x in validationInputs], validationOutputs, 'g^', label='validation data')
# plt.xlabel('Freedom')
# plt.ylabel('happiness')
# plt.title('Freedom vs. happiness')
# plt.legend()

# plt.tight_layout()
# plt.show()


# learning step: init and train a linear regression model y = f(x) = w0 + w1 * x1 + w2 * x2
# Prediction step: used the trained model to estimate the output for a new input

# # using developed code
# from utils.functions import MyLinearBivariateRegression

# # model initialisation
# regressor = MyLinearBivariateRegression()
# # training the model by using the training inputs and known training outputs
# regressor.fit(trainInputs, trainOutputs)
# # save the model parameters
# w0 = regressor.intercept_
# w1, w2 = regressor.coef_
# print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1 + ', w2, ' * x2')


# using sklearn 
# training data preparation (the sklearn linear model requires as input training data as noSamples x noFeatures array; in the current case, the input must be a matrix of len(trainInputs) lineas and two columns (two features are used in this problem))
xx = [[el1, el2] for el1, el2 in trainInputs]

# # model initialisation
regressor = linear_model.LinearRegression()
# # training the model by using the training inputs and known training outputs
regressor.fit(trainInputs, trainOutputs)
# # save the model parameters
w0 = regressor.intercept_
w1, w2 = regressor.coef_
print('the learnt model: f(x) = ', w0, ' + ', w1, ' * x1 + ', w2, ' * x2')


# plot the learnt model
# prepare some synthetic data for visualization
# create a mesh grid for GDP and Freedom values
gdp_min, gdp_max = min(x[0] for x in trainInputs), max(x[0] for x in trainInputs)
freedom_min, freedom_max = min(x[1] for x in trainInputs), max(x[1] for x in trainInputs)
gdp_range = np.linspace(gdp_min, gdp_max, 20)
freedom_range = np.linspace(freedom_min, freedom_max, 20)
gdp_mesh, freedom_mesh = np.meshgrid(gdp_range, freedom_range)

# Calculate predicted happiness for each point in the mesh
Z = w0 + w1 * gdp_mesh + w2 * freedom_mesh

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the training points
ax.scatter([x[0] for x in trainInputs], [x[1] for x in trainInputs], trainOutputs, 
           c='r', marker='o', label='Training Data')

# Plot the regression plane
surface = ax.plot_surface(gdp_mesh, freedom_mesh, Z, alpha=0.3, cmap='viridis')

# Labels and title
ax.set_xlabel('GDP capita')
ax.set_ylabel('Freedom')
ax.set_zlabel('Happiness')
ax.set_title('Bivariate Regression: GDP & Freedom vs Happiness')

# Add a color bar
fig.colorbar(surface)

plt.legend()
plt.show()



# use the trained model to predict new inputs

# makes predictions for validation data (by tool)
computedValidationOutputs = regressor.predict(validationInputs)

# Convert validation inputs to numpy array for easier handling
validationInputs = np.array(validationInputs)
computedValidationOutputs = np.array(computedValidationOutputs)

# Ensure computedValidationOutputs is a 1D array
computedValidationOutputs = np.array(computedValidationOutputs).flatten()

# # Create figure for validation results
# plt.figure(figsize=(12, 5))

# # First subplot - GDP vs Happiness
# plt.subplot(1, 2, 1)
# plt.scatter(validationInputs[:, 0], validationOutputs, marker='^', c='g', label='real validation data')
# plt.scatter(validationInputs[:, 0], computedValidationOutputs, marker='o', c='y', label='computed validation data')
# plt.xlabel('GDP capita')
# plt.ylabel('happiness')
# plt.title('GDP capita vs. happiness (validation)')
# plt.legend()

# # Second subplot - Freedom vs Happiness
# plt.subplot(1, 2, 2)
# plt.scatter(validationInputs[:, 1], validationOutputs, marker='^', c='g', label='real validation data')
# plt.scatter(validationInputs[:, 1], computedValidationOutputs, marker='o', c='y', label='computed validation data')
# plt.xlabel('Freedom')
# plt.ylabel('happiness')
# plt.title('Freedom vs. happiness (validation)')
# plt.legend()

# plt.tight_layout()
# plt.show()

# compute the prediction error
error = 0.0
for real, computed in zip(validationOutputs, computedValidationOutputs):
    error += (real - computed) ** 2
error = error / len(validationOutputs)
print('prediction error (manual): ', error)

# compute the prediction error using sklearn
from sklearn.metrics import mean_squared_error
error_sklearn = mean_squared_error(validationOutputs, computedValidationOutputs)
print('prediction error (tool): ', error_sklearn)



