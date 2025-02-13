#!/usr/bin/python
import argparse
import os
import numpy as np
import time
import sys
import shutil
sys.path.append("../../WFsim")
from wfsim import run_simulation
from sklearn.ensemble import RandomForestRegressor
from statistics import statisticsClass
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import multiprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import concurrent.futures
from nn import PopulationGeneticsModel
import torch

NUMBER_OF_STATISTICS = 5
t = 30
DEBUG = 0  ## BOUCHER: Change this to 1 for debuggin mode
OUTPUTFILENAME = "priors.txt"

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
directory = "temp"
path = os.path.join("./", directory)

POPULATION_GENERATOR = "./build/OneSamp"

def getName(filename):
    (_, filename) = os.path.split(filename)
    return filename


#############################################################
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=float, help="Minimum Allele Frequency")
parser.add_argument("--r", type=float, help="Mutation Rate")
parser.add_argument("--lNe", type=int, help="Lower of Ne Range")
parser.add_argument("--uNe", type=int, help="Upper of Ne Range")
parser.add_argument("--lT", type=float, help="Lower of Theta Range")
parser.add_argument("--uT", type=float, help="Upper of Theta Range")
parser.add_argument("--s", type=int, help="Number of OneSamp Trials")
parser.add_argument("--lD", type=float, help="Lower of Duration Range")
parser.add_argument("--uD", type=float, help="Upper of Duration Range")
# parser.add_argument("--i", type=float, help="Missing data for individuals")
# parser.add_argument("--l", type=float, help="Missing data for loci")
parser.add_argument("--o", type=str, help="The File Name")
parser.add_argument("--t", type=int, help="Repeat times")
parser.add_argument("--n", type=bool, help="whether to filter the monomorphic loci", default=False)
# parser.add_argument("--md", type=str, help="Model Name")

args = parser.parse_args()

#########################################
# INITIALIZING PARAMETERS
#########################################
if (args.t):
    t = int(args.t)

minAlleleFreq = 0.05
if (args.m):
    minAlleleFreq = float(args.m)

mutationRate = 0.000000012

if (args.r):
    mutationRate = float(args.r)

lowerNe = 150
if (args.lNe):
    lowerNe = int(args.lNe)

upperNe = 250
if (args.uNe):
    upperNe = int(args.uNe)

if (int(lowerNe) > int(upperNe)):
    print("ERROR:main:lowerNe > upperNe. Fatal Error")
    exit()

if (int(lowerNe) < 1):
    print("ERROR:main:lowerNe must be a positive value. Fatal Error")
    exit()

if (int(upperNe) < 1):
    print("ERROR:main:upperNe must be a positive value. Fatal Error")
    exit()

rangeNe = "%d,%d" % (lowerNe, upperNe)

lowerTheta = 0.000048
if (args.lT):
    lowerTheta = float(args.lT)

upperTheta = 0.0048
if (args.uT):
    upperTheta = float(args.uT)

rangeTheta = "%f,%f" % (lowerTheta, upperTheta)

numOneSampTrials = 20
if (args.s):
    numOneSampTrials = int(args.s)

lowerDuration = 2
if (args.lD):
    lowerDuration = float(args.lD)

upperDuration = 8
if (args.uD):
    upperDuration = float(args.uD)

# indivMissing = .2
# if (args.i):
#     indivMissing = float(args.i)
#
# lociMissing = .2
# if (args.l):
#     lociMissing = float(args.l)

rangeDuration = "%f,%f" % (lowerDuration, upperDuration)

# fileName = "oneSampIn"
fileName = "data/genePop50x80"
if (args.o):
    fileName = str(args.o)
else:
    print("WARNING:main: No filename provided.  Using oneSampIn")

if (DEBUG):
    print("Start calculation of statistics for input population")

rangeTheta = "%f,%f" % (lowerTheta, upperTheta)



#########################################
# STARTING INITIAL POPULATION
#########################################

inputFileStatistics = statisticsClass()

inputFileStatistics.readData(fileName)
# inputFileStatistics.filterIndividuals(indivMissing)
# inputFileStatistics.filterLoci(lociMissing)
if (args.n):
    inputFileStatistics.filterMonomorphicLoci()

inputFileStatistics.test_stat1()
inputFileStatistics.test_stat2()
inputFileStatistics.test_stat3()
inputFileStatistics.test_stat5()
inputFileStatistics.test_stat4()

numLoci = inputFileStatistics.numLoci
sampleSize = inputFileStatistics.sampleSize

##Creating input file & List with intial statistics
textList = [str(inputFileStatistics.stat1), str(inputFileStatistics.stat2), str(inputFileStatistics.stat3),
             str(inputFileStatistics.stat4), str(inputFileStatistics.stat5)]
inputStatsList = textList

inputPopStats = "inputPopStats_" + getName(fileName) + "_" + str(t)
with open(inputPopStats, 'w') as fileINPUT:
    fileINPUT.write('\t'.join(textList[0:]) + '\t')
fileINPUT.close()

if (DEBUG):
    print("Finish calculation of statistics for input population")

#############################################
# FINISH STATS FOR INITIAL INPUT  POPULATION
############################################

#########################################
# STARTING ALL POPULATIONS
#########################################
#Result queue
results_list = []

if (DEBUG):
    print("Start calculation of statistics for ALL populations")

statistics1 = []
statistics2 = []
statistics3 = []
statistics4 = []
statistics5 = []

statistics1 = [0 for x in range(numOneSampTrials)]
statistics2 = [0 for x in range(numOneSampTrials)]
statistics3 = [0 for x in range(numOneSampTrials)]
statistics5 = [0 for x in range(numOneSampTrials)]
statistics4 = [0 for x in range(numOneSampTrials)]


# Generate random populations and calculate summary statistics
def processRandomPopulation(x):
    loci = inputFileStatistics.numLoci
    sampleSize = inputFileStatistics.sampleSize
    proc = multiprocessing.Process()
    process_id = os.getpid()
    # change the intermediate file name by process id
    intermediateFilename = str(process_id) + "_intermediate_" + getName(fileName) + "_" + str(t)
    intermediateFile = os.path.join(path, intermediateFilename)
    Ne_left = lowerNe
    Ne_right = upperNe
    if Ne_left % 2 != 0:
        Ne_left += 1
    num_evens = (Ne_right - Ne_left) // 2 + 1
    random_index = random.randint(0, num_evens - 1)
    target_Ne = Ne_left + random_index * 2
    target_Ne = f"{target_Ne:05d}"
    run_simulation(minAlleleFreq, mutationRate,  lowerNe, upperNe, lowerTheta, upperTheta, lowerDuration, upperDuration, loci, sampleSize, intermediateFile)

    # if (DEBUG):
    #     print(cmd)
    #
    # returned_value = os.system(cmd)
    #
    # if returned_value:
    #     print("ERROR:main:Refactor did not run")


    refactorFileStatistics = statisticsClass()

    refactorFileStatistics.readData(intermediateFile)
    # refactorFileStatistics.filterIndividuals(indivMissing)
    # refactorFileStatistics.filterLoci(lociMissing)
    refactorFileStatistics.test_stat1()
    refactorFileStatistics.test_stat2()
    refactorFileStatistics.test_stat3()
    refactorFileStatistics.test_stat5()
    refactorFileStatistics.test_stat4()

    statistics1[x] = refactorFileStatistics.stat1
    statistics2[x] = refactorFileStatistics.stat2
    statistics3[x] = refactorFileStatistics.stat3
    statistics5[x] = refactorFileStatistics.stat5
    statistics4[x] = refactorFileStatistics.stat4

    # Making file with stats from all populations
    textList = [str(refactorFileStatistics.NE_VALUE), str(refactorFileStatistics.stat1),
                str(refactorFileStatistics.stat2),
                str(refactorFileStatistics.stat3),
                str(refactorFileStatistics.stat4), str(refactorFileStatistics.stat5)]

    return textList


try:
    os.mkdir(path)
except FileExistsError:
    pass

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')
    # Parallel process the random populations and add to a list
    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        for result in executor.map(processRandomPopulation, range(numOneSampTrials)):
            try:
                results_list.append(result)
            except Exception as e:
                print(f"Generated an exception: {e}")


try:
    shutil.rmtree(directory)
except FileNotFoundError:
    print(f"Directory '{directory}' not found.")

########################################
# FINISHING ALL POPULATIONS
########################################

########################################
# Neural Network
########################################

# Assign input and all population stats to dataframes with column names
allPopStatistics = pd.DataFrame(results_list, columns=['Ne', 'Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium'])
inputStatsList = pd.DataFrame([inputStatsList], columns=['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium'])

# Assign dependent and independent variables for regression model
Z = np.array(inputStatsList[['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium']])
X = np.array(allPopStatistics[['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium']])
y = np.array(allPopStatistics['Ne'])
y = np.array([float(value) for value in y if float(value) > 0])

# #Normalize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# Z_scaled = scaler.fit_transform(Z)
#
# #Apply box-cox transformation
# y_transformed, lambda_value = boxcox(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Convert to PyTorch tensors
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = y_train.astype(np.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
y_test = y_test.astype(np.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)
Z = Z.astype(np.float32)
Z = torch.tensor(Z, dtype=torch.float32).to(device)


print(f"\n-----------------Neural Network------------------")

pop_gen_model = PopulationGeneticsModel(learning_rate=0.001, epochs=100, batch_size=128)
pop_gen_model.train(X_train, y_train, X_test, y_test)

prediction_results = pop_gen_model.predict_with_uncertainty(Z, n_simulations=100)
print("Prediction Results")
print(prediction_results)

evaluation_results = pop_gen_model.evaluate(X_test, y_test)
print(" ")
print(evaluation_results)





# ##########################
# # RANDOM FOREST REGRESSION
# ##########################
print(f"\n-----------------RANDOM FOREST------------------")

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=80, random_state=42)
y_train = y_train.ravel()

rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)

# Predict the Ne value for input population
rf_prediction = rf_regressor.predict(Z)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# print(f"\nPrediction:")
# print(rf_prediction.round(decimals=2))

# Calculate confidence interval, Get the predictions from each tree for the new data point
tree_predictions = np.array([tree.predict(Z) for tree in rf_regressor.estimators_])

median_prediction = np.median(tree_predictions)
mean_prediction = np.mean(tree_predictions)
min_prediction = np.min(tree_predictions)
max_prediction = np.max(tree_predictions)

# Calculate the 2.5th and 97.5th percentiles for the 95% confidence interval
lower_bound = np.percentile(tree_predictions, 2.5)
upper_bound = np.percentile(tree_predictions, 97.5)

print("Prediction Results")
print(f"{min_prediction} {max_prediction} {mean_prediction} {median_prediction} {lower_bound} {upper_bound}")
print(" ")
print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

'''
# Get numerical feature importances
importances = list(rf_regressor.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(inputStatsList.columns, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Print out the feature and importances
print("\nFeature importance")
[print('Variable: {:30} : {}'.format(*pair)) for pair in feature_importances]
'''

print("")
print("----- %s seconds -----" % (time.time() - start_time))





# # Fit the linear regression model
# model = LinearRegression()
# result = model.fit(X_train, y_train)
#
# print(f"\n-----------------LINEAR REGRESSION------------------")
#
# #Predict for Test values
# y_pred = result.predict(X_test)
# '''
# # Calculate errors
# absolute_errors = np.abs(y_pred - y_test)
# min = np.min(absolute_errors)
# max = np.max(absolute_errors)
# q1 = np.percentile(absolute_errors, 25)
# median = np.percentile(absolute_errors, 50)
# q3 = np.percentile(absolute_errors, 75)
# mae = np.mean(absolute_errors)
#
# # Compute MSE, RMSE, and MAE for the test set
# mse_test = mean_squared_error(y_test, y_pred)
# rmse_test = np.sqrt(mse_test)
# mae_test = mean_absolute_error(y_test, y_pred)
# print(f"MSE: {mse_test:.2f}")
# print(f"RMSE: {rmse_test:.2f}")
# print(f"MAE: {mae_test:.2f}")
# '''
# print(f"{min:.2f} {max:.2f} {median:.2f} {q1:.2f} {q3:.2f}")
#
# # Predict the value for the query point
# # prediction = model.predict(Z_scaled)
# prediction = model.predict(Z)
# # y_original_scale = inv_boxcox(prediction, lambda_value)
# # print("\n Effective population for input population:", y_original_scale[0])
#
# ####### CALCULATING CONFIDENCE INTERVAL #########
#
# def getConfInterval(model, X_train, y_train, Z, prediction):
#     # Convert to a numeric array
#     X_train = X_train.astype(np.float64)
#     y_train = y_train.astype(np.float64)
#     Z = Z.astype(np.float64)
#
#     # Predictions on the training data
#     y_train_pred = model.predict(X_train)
#     # MSE on the training data
#     mse = np.mean((y_train - y_train_pred) ** 2)
#     # compute the (X'X)^-1 matrix
#     XX_inv = np.linalg.inv(np.dot(X_train.T, X_train))
#     # compute the leverage (hat) matrix for the new data point
#     hat_matrix = np.dot(np.dot(Z, XX_inv), Z.T)
#     # calculate the standard error of the prediction
#     std_error = np.sqrt((1 + hat_matrix) * mse)
#     # The t value for the 95% confidence interval
#     t_value = stats.t.ppf(1 - 0.05 / 2, df=len(X_train) - X_train.shape[1] - 1)
#
#     # Confidence interval for the new prediction
#     ci_lower = prediction - t_value * std_error
#     ci_upper = prediction + t_value * std_error
#
#     print(f"95% confidence interval: [{ci_lower[0][0].round(decimals=2)}, {ci_upper[0][0].round(decimals=2)}]")
#
# # Output the result
# print(f"\nPrediction: {prediction.round(decimals=2)}")
# getConfInterval(model, X_train, y_train, Z, prediction)
#
# '''
# # Get the coefficients for each feature
# coefficients = model.coef_
# # coefficients_original_scale = coefficients / lambda_value
#
# # Print the coefficients for each feature
# print("\nCoefficients for each feature:")
# for feature, coef in zip(inputStatsList.columns, coefficients):
#     print(f"Variable: {feature}: {coef:.2f}")
# '''
# # Perform k-fold cross-validation
# # cv_scores = cross_val_score(model, X_scaled, y_transformed, cv=10)
# # cv_scores = cross_val_score(model, X, y, cv=10)
# # print("\nCross validation scores : ", round(cv_scores[0],2), round(cv_scores[1],2), round(cv_scores[2],2), round(cv_scores[3],2), round(cv_scores[4],2))
#
# print("----- %s seconds -----" % (time.time() - start_time))


