#!/usr/bin/python
import argparse
import os
import numpy as np
import time
import sys
import shutil
sys.path.append("/blue/boucher/suhashidesilva/2025/WFsim")
from wfsim import run_simulation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor, plot_importance
from statistics import statisticsClass
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import multiprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import concurrent.futures
import torch
from sklearn.preprocessing import StandardScaler
from nn import PopulationGeneticsModel, train_xgboost, ensemble_predict


# print(torch.__version__)  # PyTorch version
# print(torch.version.cuda)
# print(torch.cuda.is_available())

NUMBER_OF_STATISTICS = 5
t = 30
DEBUG = 0  ## BOUCHER: Change this to 1 for debuggin mode
OUTPUTFILENAME = "priors.txt"

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
directory = "temp"
path = os.path.join("./", directory)
results_path = "/blue/boucher/suhashidesilva/2025/ONeSAMP_3.1/output/V30/"

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
#parser.add_argument("--i", type=float, help="Missing data for individuals")
#parser.add_argument("--l", type=float, help="Missing data for loci")
parser.add_argument("--o", type=str, help="The File Name")
#parser.add_argument("--t", type=int, help="Repeat times")
#parser.add_argument("--n", type=bool, help="whether to filter the monomorphic loci", default=False)
# parser.add_argument("--md", type=str, help="Model Name")

args = parser.parse_args()

#########################################
# INITIALIZING PARAMETERS
#########################################
#if (args.t):
#    t = int(args.t)

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

numOneSampTrials = 20000
if (args.s):
    numOneSampTrials = int(args.s)

lowerDuration = 2
if (args.lD):
    lowerDuration = float(args.lD)

upperDuration = 8
if (args.uD):
    upperDuration = float(args.uD)

#indivMissing = .2
#if (args.i):
#    indivMissing = float(args.i)

#lociMissing = .2
#if (args.l):
#    lociMissing = float(args.l)

rangeDuration = "%f,%f" % (lowerDuration, upperDuration)

# fileName = "oneSampIn"
#fileName = "data/genePop50x80"
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
#if (args.n):
#    inputFileStatistics.filterMonomorphicLoci()

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

inputPopStats = results_path + "inputPopStats_" + getName(fileName)
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
    #cmd = "%s -u%.9f -v%s -rC -l%d -i%d -d%s -s -t1 -b%s -f%f -o1 -p > %s" % (POPULATION_GENERATOR, mutationRate, rangeTheta, loci, sampleSize, rangeDuration, target_Ne, minAlleleFreq, intermediateFile)
    #print(minAlleleFreq, mutationRate,  lowerNe, upperNe, lowerTheta, upperTheta, lowerDuration, upperDuration, loci, sampleSize, intermediateFile)
    run_simulation(minAlleleFreq, mutationRate,  lowerNe, upperNe, lowerTheta, upperTheta, lowerDuration, upperDuration, loci, sampleSize, intermediateFile)
    '''
    if (DEBUG):
        print(cmd)

    returned_value = os.system(cmd)
    
    if returned_value:
        print("ERROR:main:Refactor did not run")
    '''

    refactorFileStatistics = statisticsClass()

    refactorFileStatistics.readData(intermediateFile)
    # refactorFileStatistics.filterIndividuals(indivMissing)
    # refactorFileStatistics.filterLoci(lociMissing)
    refactorFileStatistics.test_stat1()
    refactorFileStatistics.test_stat2()
    refactorFileStatistics.test_stat3()
    refactorFileStatistics.test_stat4()
    refactorFileStatistics.test_stat5()

    statistics1[x] = refactorFileStatistics.stat1
    statistics2[x] = refactorFileStatistics.stat2
    statistics3[x] = refactorFileStatistics.stat3
    statistics4[x] = refactorFileStatistics.stat4
    statistics5[x] = refactorFileStatistics.stat5


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
allPopStatistics = pd.DataFrame(results_list, columns=['Ne', 'Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_equilibrium'])
inputStatsList = pd.DataFrame([inputStatsList], columns=['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_equilibrium'])

allPopStats = results_path + "allPopStats_" + getName(fileName) 
allPopStatistics.to_csv(allPopStats, index=False)



# Assign dependent and independent variables for regression model
Z = np.array(inputStatsList[['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_equilibrium']])
X = np.array(allPopStatistics[['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance','Gametic_equilibrium']])
y = np.array(allPopStatistics['Ne'])
y = np.array([float(value) for value in y if float(value) > 0])

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.3, random_state=40)

# #Normalize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# Z_scaled = scaler.fit_transform(Z)
#
# #Apply box-cox transformation
# y_transformed, lambda_value = boxcox(y)
# ##########################
# # RANDOM FOREST REGRESSION
# ##########################

print(f"\n-----------------RANDOM FOREST------------")

# Define the hyperparameter search space
param_dist = {
    'n_estimators': [1000],
    'max_depth': [40, 80],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True]
}

# Initialize the base model
rf = RandomForestRegressor(random_state=42)

# Set up the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,                 # You can increase this for deeper search
    cv=3,                      # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1,                 # Use all cores
    scoring='neg_mean_squared_error'
)

# Perform the search
random_search.fit(X_train_np, y_train_np.ravel())

# Extract the best estimator
best_rf = random_search.best_estimator_

# Predict using the best model
y_pred = best_rf.predict(X_test_np)

# Predict the Ne value for input population
rf_prediction = best_rf.predict(Z)

mse = mean_squared_error(y_test_np, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_np, y_pred)

# Calculate confidence interval, Get the predictions from each tree for the new data point
tree_predictions = np.array([tree.predict(Z) for tree in best_rf.estimators_])

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

# Get numerical feature importances
importances = list(best_rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(inputStatsList.columns, importances)]

# Sort the feature importances
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print("\nFeature importance")
[print('Variable: {:30} : {}'.format(*pair)) for pair in feature_importances]


'''

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#x_scaler = StandardScaler()
#y_scaler = StandardScaler()

#X_train = x_scaler.fit_transform(X_train)
#X_test = x_scaler.transform(X_test)  # Use same scaler on test data
#Z = x_scaler.transform(Z) 

#y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
#y_test = y_scaler.transform(y_test.reshape(-1, 1))

# X_train = torch.tensor(X_train_np.astype(np.float32), dtype=torch.float32).to(device)
# X_test = torch.tensor(X_test_np.astype(np.float32), dtype=torch.float32).to(device)
# y_train = torch.tensor(y_train_np.astype(np.float32).reshape(-1, 1), dtype=torch.float32).to(device)
# y_test = torch.tensor(y_test_np.astype(np.float32).reshape(-1, 1), dtype=torch.float32).to(device)
# Z_tensor = torch.tensor(Z.astype(np.float32), dtype=torch.float32).to(device)

# ---- Train models ----
print(f"\n-----------------Neural Network------------------")
nn_model = PopulationGeneticsModel(input_size=X.shape[1])
nn_model.train(X_train, y_train, X_test, y_test)
nn_pred = nn_model.predict(X_test)
'''

print(f"\n-----------------XGBoost------------------")


# Initialize the XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Flatten y just in case
y_train = y_train_np.ravel()

# Train the model
xgb_model.fit(X_train_np, y_train)

# Predict on test data
y_pred_xgb = xgb_model.predict(X_test_np)

# Predict on new data (Z)
xgb_prediction = xgb_model.predict(Z)

# Manual tree-by-tree predictions to estimate uncertainty
tree_preds = np.array([
    xgb_model.predict(Z, iteration_range=(i, i+1))
    for i in range(xgb_model.get_booster().num_boosted_rounds())
]).reshape(-1)

# Compute statistics
min_pred = np.min(tree_preds)
max_pred = np.max(tree_preds)
mean_pred = np.mean(tree_preds)
median_pred = np.median(tree_preds)
lower_bound = np.percentile(tree_preds, 2.5)
upper_bound = np.percentile(tree_preds, 97.5)

# Report
print("\nXGBoost Prediction on Z (Single Input)")
print(f"Min:          {min_pred:.4f}")
print(f"Max:          {max_pred:.4f}")
print(f"Mean:         {mean_pred:.4f}")
print(f"Median:       {median_pred:.4f}")
print(f"95% CI:       [{lower_bound:.4f}, {upper_bound:.4f}]")

# Evaluate performance
mse_xgb = mean_squared_error(y_test_np, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test_np, y_pred_xgb)

print("XGBoost Prediction Metrics")
print(f"MSE: {mse_xgb:.2f}, RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}")
print(f"\nXGBoost Prediction on Z: {xgb_prediction.round(2)}")

# -----------------------------
# Feature Importances (Numerical)
# -----------------------------
importances = xgb_model.feature_importances_
feature_importances = [(feature, round(score, 2)) for feature, score in zip(inputStatsList.columns, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print("\nXGBoost Feature Importances:")
[print(f"Variable: {f:30} | Importance: {imp}") for f, imp in feature_importances]

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


