#!/usr/bin/python
import argparse
import os
import numpy as np
import pandas as pd
import time
import random
import multiprocessing
import concurrent.futures
import sys
import shutil
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.linear_model import Ridge, Lasso
from statistics import statisticsClass
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
import torch
#sys.path.append("/blue/boucher/suhashidesilva/2025/WFsim")
from wfsim import run_simulation


# print(torch.__version__)  # PyTorch version
# print(torch.version.cuda)
# print(torch.cuda.is_available())

NUMBER_OF_STATISTICS = 5
t = 1
DEBUG = 0  ## BOUCHER: Change this to 1 for debuggin mode
OUTPUTFILENAME = "priors.txt"

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
directory = "temp"
path = os.path.join("../", directory)
#results_path = "/blue/boucher/suhashidesilva/2025/ONeSAMP_3.1_V1/output_100_1/"

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
parser.add_argument("--i", type=float, help="Missing data for individuals")
parser.add_argument("--l", type=float, help="Missing data for loci")
parser.add_argument("--o", type=str, help="The File Name")
parser.add_argument("--t", type=int, help="Repeat times")
parser.add_argument("--n", type=bool, help="whether to filter the monomorphic loci", default=False)
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

lowerNe = 4
if (args.lNe):
    lowerNe = int(args.lNe)

upperNe = 200
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

indivMissing = .2
if (args.i):
    indivMissing = float(args.i)

lociMissing = .2
if (args.l):
    lociMissing = float(args.l)

rangeDuration = "%f,%f" % (lowerDuration, upperDuration)

fileName = "oneSampIn"

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
inputFileStatistics.filterIndividuals(indivMissing)
inputFileStatistics.filterLoci(lociMissing)
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

'''
inputPopStats = results_path + "inputPopStats_" + getName(fileName)
with open(inputPopStats, 'w') as fileINPUT:
    fileINPUT.write('\t'.join(textList[0:]) + '\t')
fileINPUT.close()
'''

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
    cmd = "%s -u%.9f -v%s -rC -l%d -i%d -d%s -s -t1 -b%s -f%f -o1 -p > %s" % (POPULATION_GENERATOR, mutationRate, rangeTheta, loci, sampleSize, rangeDuration, target_Ne, minAlleleFreq, intermediateFile)
    #print(minAlleleFreq, mutationRate,  lowerNe, upperNe, lowerTheta, upperTheta, lowerDuration, upperDuration, loci, sampleSize, intermediateFile)
    #run_simulation(minAlleleFreq, mutationRate,  lowerNe, upperNe, lowerTheta, upperTheta, lowerDuration, upperDuration, loci, sampleSize, intermediateFile)

    if (DEBUG):
        print(cmd)

    returned_value = os.system(cmd)

    if returned_value:
        print("ERROR:main:Refactor did not run")


    refactorFileStatistics = statisticsClass()

    refactorFileStatistics.readData(intermediateFile)
    refactorFileStatistics.filterIndividuals(indivMissing)
    refactorFileStatistics.filterLoci(lociMissing)
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
allPopStatistics = pd.DataFrame(results_list, columns=['Ne','Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt'])
inputStatsList = pd.DataFrame([inputStatsList], columns=['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt'])

'''
allPopStats = results_path + "allPopStats_" + getName(fileName) 
allPopStatistics.to_csv(allPopStats, index=False)
'''

# Assign dependent and independent variables for regression model
Z = np.array(inputStatsList[['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']].astype(float).to_numpy())
X = np.array(allPopStatistics[['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']].astype(float).to_numpy())
y = np.array(allPopStatistics['Ne'])
y = np.array([float(value) for value in y if float(value) > 0])

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.3, random_state=40)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.fit_transform(X_test_np)
Z_scaled = scaler.transform(Z)

def evaluate_cv(model, X, y, cv_folds=5):
    r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    rmse_scores = -cross_val_score(model, X, y, cv=cv_folds,
        scoring=make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False))
    mae_scores = -cross_val_score(model, X, y, cv=cv_folds,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False))

    print(f"{model.__class__.__name__} ({cv_folds}-Fold CV) => R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}, "
      f"RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}, "
      f"MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")




# -------------------------------
# Random Forest Regression
# -------------------------------

loci = inputFileStatistics.numLoci
sampleSize = inputFileStatistics.sampleSize
output_path = (f"/blue/boucher/suhashidesilva/2025/ONeSAMP_3.1_V1/output_100_2/samples/genePop{sampleSize}x{loci}")
os.makedirs(output_path, exist_ok=True)

print(f"\n-----------------RANDOM FOREST------------")

# --- Train Random Forest without tuning ---
rf_model = RandomForestRegressor(
    n_estimators=5000,
    max_depth=80,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features='log2',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train_np.ravel())

rf_path = os.path.join(output_path, f"rf_model_{sampleSize}x{loci}.joblib")
scalar_path = os.path.join(output_path, f"scaler_{sampleSize}x{loci}.joblib")

# --- Save the trained model and scaler ---
joblib.dump(rf_model, rf_path)
joblib.dump(scaler, scalar_path)

# --- Load the model and scaler (for future use) ---
# rf_model = joblib.load(os.path.join(output_path, f"rf_model_{sampleSize}x{loci}.joblib"))
# scaler = joblib.load(os.path.join(output_path, f'scaler_{sampleSize}x{loci}.joblib'))
# Z_scaled = scaler.transform(Z)

# --- Make predictions ---
y_pred = rf_model.predict(X_test_scaled)
rf_prediction = rf_model.predict(Z_scaled)

# --- Evaluate performance ---
mse = mean_squared_error(y_test_np, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_np, y_pred)
r2 = r2_score(y_test_np, y_pred)

# --- Confidence Interval from all trees ---
tree_predictions = np.array([tree.predict(Z_scaled) for tree in rf_model.estimators_])
median_prediction = np.median(tree_predictions)
mean_prediction = np.mean(tree_predictions)
min_prediction = np.min(tree_predictions)
max_prediction = np.max(tree_predictions)
lower_bound = np.percentile(tree_predictions, 2.5)
upper_bound = np.percentile(tree_predictions, 97.5)

# --- Print results ---
print("Prediction Results")
print(f"Min: {min_prediction:.2f}, Max: {max_prediction:.2f}, Mean: {mean_prediction:.2f}, Median: {median_prediction:.2f}, 95% CI: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

# --- Feature importances ---
importances = rf_model.feature_importances_
feature_importances = [(feature, round(score, 2)) for feature, score in zip(inputStatsList.columns, importances)]
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
# -------------------------------
# XGBoost
# -------------------------------

print(f"\n-----------------XGBoost------------------")

# --- Define fixed parameters for XGBoost ---
xgb_model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    min_child_weight=3,
    subsample=0.6,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# --- Train the model ---
xgb_model.fit(X_train_scaled, y_train_np.ravel())

# --- Save the trained model and scaler ---
xgb_path = os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib")

joblib.dump(xgb_model, xgb_path)

'''
# Load model and scaler
xgb_model = joblib.load(os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib"))

# Predict again using new Z
Z_scaled = scaler.transform(Z)
xgb_prediction = xgb_model.predict(Z_scaled)
'''

# --- Predict on test set and input population Z ---
y_pred_xgb = xgb_model.predict(X_test_scaled)
xgb_prediction = xgb_model.predict(Z_scaled)

# --- Estimate prediction uncertainty using per-tree predictions ---
tree_preds = np.array([
    xgb_model.predict(Z_scaled, iteration_range=(i, i+1))
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
print(f"Min: {min_pred:.4f}, Max: {max_pred:.4f}, Mean: {mean_pred:.4f}, Median: {median_pred:.4f}, 95% CI: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Evaluate performance
mse_xgb = mean_squared_error(y_test_np, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test_np, y_pred_xgb)
r2_xgb = r2_score(y_test_np, y_pred_xgb)

print("\nXGBoost Prediction Metrics")
print(f"MSE: {mse_xgb:.2f}, RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}, R2: {r2_xgb:.2f}")
#print(f"\nXGBoost Prediction on Z: {xgb_prediction_bc.round(2)}")

# Feature Importances
importances = xgb_model.feature_importances_
feature_importances = [(feature, round(score, 2)) for feature, score in zip(inputStatsList.columns, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print("\nXGBoost Feature Importances:")
[print(f"Variable: {f:30} | {imp}") for f, imp in feature_importances]

print("")

# -------------------------------
# Lasso & Ridge Regression
# -------------------------------

print(f"\n-----------------Lasso & Ridge Regression------------------")

X_train_path = os.path.join(output_path, f"X_train_scaled_{sampleSize}x{loci}.joblib")
y_train_path = os.path.join(output_path, f"y_train_{sampleSize}x{loci}.joblib")

joblib.dump(X_train_scaled, X_train_path)
joblib.dump(y_train_np, y_train_path)

ridge_model = Ridge(alpha=10)
ridge_model.fit(X_train_scaled, y_train_np)

ridge_path = os.path.join(output_path, f"ridge_model_{sampleSize}x{loci}.joblib")
joblib.dump(ridge_model, ridge_path)

lasso_model = Lasso(alpha=0.01, max_iter=10000)
lasso_model.fit(X_train_scaled, y_train_np)

lasso_path = os.path.join(output_path, f"lasso_model_{sampleSize}x{loci}.joblib")
joblib.dump(lasso_model, lasso_path)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

# ---- Bootstrap-Based Prediction Statistics ---- #
def predict_with_stats(model, X_train, y_train, new_sample, n_bootstrap=1000, alpha=0.05):
    predictions = []
    for _ in range(n_bootstrap):
        X_boot, y_boot = resample(X_train, y_train)
        model.fit(X_boot, y_boot)
        pred = model.predict(new_sample)[0]
        predictions.append(pred)
    predictions = np.array(predictions)
    lower = np.percentile(predictions, 100 * (alpha / 2))
    upper = np.percentile(predictions, 100 * (1 - alpha / 2))
    return {
        'mean': np.mean(predictions),
        'median': np.median(predictions),
        'min': np.min(predictions),
        'max': np.max(predictions),
        '95% CI': (lower, upper)
    }

def print_stats_inline(name, stats):
    print(f"{name} => Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, "
          f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}, "
          f"95% CI: ({stats['95% CI'][0]:.4f}, {stats['95% CI'][1]:.4f})")

feature_names = ['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']

ridge_loaded = joblib.load(os.path.join(output_path, f'ridge_model_{sampleSize}x{loci}.joblib'))
lasso_loaded = joblib.load(os.path.join(output_path, f'lasso_model_{sampleSize}x{loci}.joblib'))

ridge_coef = ridge_model.coef_
lasso_coef = lasso_model.coef_

ridge_rmse, ridge_mae, ridge_r2 = evaluate_model(ridge_loaded, X_test_scaled, y_test_np)
lasso_rmse, lasso_mae, lasso_r2 = evaluate_model(lasso_loaded, X_test_scaled, y_test_np)

ridge_stats = predict_with_stats(ridge_loaded, X_train_scaled, y_train_np, Z_scaled)
lasso_stats = predict_with_stats(lasso_loaded, X_train_scaled, y_train_np, Z_scaled)

print(f"Ridge => RMSE: {ridge_rmse:.4f}, MAE: {ridge_mae:.4f}, R2: {ridge_r2:.4f}")
print_stats_inline("Ridge", ridge_stats)

print("\nRidge Feature Importances:")
for name, coef in sorted(zip(feature_names, ridge_coef), key=lambda x: abs(x[1]), reverse=True):
    print(f"{name}: {coef:.4f}")

print(f"\nLasso => RMSE: {lasso_rmse:.4f}, MAE: {lasso_mae:.4f}, R2: {lasso_r2:.4f}")
print_stats_inline("Lasso", lasso_stats)

print("\nLasso Feature Importances:")
for name, coef in sorted(zip(feature_names, lasso_coef), key=lambda x: abs(x[1]), reverse=True):
    print(f"{name}: {coef:.4f}")

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


