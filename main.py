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
# import torch
#sys.path.append("/blue/boucher/suhashidesilva/2025/WFsim")
#from wfsim import run_simulation


# print(torch.__version__)  # PyTorch version
# print(torch.version.cuda)
# print(torch.cuda.is_available())

NUMBER_OF_STATISTICS = 5
t = 1
DEBUG = 0  ## BOUCHER: Change this to 1 for debuggin mode
OUTPUTFILENAME = "priors.txt"

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
directory = "/blue/boucher/suhashidesilva/2025/ONeSAMP_3.1_V1/temp"
#directory = "temp"
path = os.path.join("/", directory)

results_path = os.path.join(BASE_PATH, "./output_70/new/")


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

lowerNe = 50
if (args.lNe):
    lowerNe = int(args.lNe)

upperNe = 150
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

#inputFileStatistics.test_stat1()
inputFileStatistics.test_stat1_new()
inputFileStatistics.test_stat2()
inputFileStatistics.test_stat3()
inputFileStatistics.test_stat5()
inputFileStatistics.test_stat4()

numLoci = inputFileStatistics.numLoci
sampleSize = inputFileStatistics.sampleSize

##Creating input file & List with intial statistics
textList = [str(inputFileStatistics.stat1_new), str(inputFileStatistics.stat2), str(inputFileStatistics.stat3),
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

#statistics1 = []
statistics1_new = []
statistics2 = []
statistics3 = []
statistics4 = []
statistics5 = []

#statistics1 = [0 for x in range(numOneSampTrials)]
statistics1_new = [0 for x in range(numOneSampTrials)]
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
    #refactorFileStatistics.test_stat1()
    refactorFileStatistics.test_stat1_new()
    refactorFileStatistics.test_stat2()
    refactorFileStatistics.test_stat3()
    refactorFileStatistics.test_stat5()
    refactorFileStatistics.test_stat4()

    #statistics1[x] = refactorFileStatistics.stat1
    statistics1_new[x] = refactorFileStatistics.stat1_new
    statistics2[x] = refactorFileStatistics.stat2
    statistics3[x] = refactorFileStatistics.stat3
    statistics5[x] = refactorFileStatistics.stat5
    statistics4[x] = refactorFileStatistics.stat4


    # Making file with stats from all populations
    textList = [str(refactorFileStatistics.NE_VALUE), str(refactorFileStatistics.stat1_new),
                str(refactorFileStatistics.stat2),
                str(refactorFileStatistics.stat3),
                str(refactorFileStatistics.stat4), str(refactorFileStatistics.stat5)]

    return textList


try:
    os.mkdir(path)
except FileExistsError:
    pass


def main():
    # Parallel process the random populations and add to a list
    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        for result in executor.map(processRandomPopulation, range(numOneSampTrials)):
            try:
                results_list.append(result)
            except Exception as e:
                print(f"Generated an exception: {e}")



if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    main()
    '''
    # Parallel process the random populations and add to a list
    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        for result in executor.map(processRandomPopulation, range(numOneSampTrials)):
            try:
                results_list.append(result)
            except Exception as e:
                print(f"Generated an exception: {e}")
    '''

try:
    shutil.rmtree(directory)
except FileNotFoundError:
    print(f"Directory '{directory}' not found.")

simulation_time = time.time()

print("----- %s seconds -----" % (time.time() - start_time))



########################################
# FINISHING ALL POPULATIONS
########################################

########################################
# Neural Network
########################################

# Assign input and all population stats to dataframes with column names
allPopStatistics = pd.DataFrame(results_list, columns=['Ne','Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt'])
inputStatsList = pd.DataFrame([inputStatsList], columns=['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt'])


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
output_path = (f"/blue/boucher/suhashidesilva/2025/ONeSAMP_3.1_V1/output_100_1L/samples/genePop{sampleSize}x{loci}")
os.makedirs(output_path, exist_ok=True)

print(f"\n-----------------RANDOM FOREST------------")

# --- Train Random Forest without tuning ---
rf_model = RandomForestRegressor(
    n_estimators=5000,
    max_depth=40,
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

evaluate_cv(rf_model, X_train_scaled, y_train_np.ravel())

# --- Feature importances ---
importances = rf_model.feature_importances_
feature_importances = [(feature, round(score, 2)) for feature, score in zip(inputStatsList.columns, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

print("\nFeature importance")
[print('Variable: {:30} : {}'.format(*pair)) for pair in feature_importances]

print("----- %s seconds -----" % (time.time() - simulation_time))

# -------------------------------
# XGBoost
# -------------------------------

print(f"\n-----------------XGBoost------------------")

base_params = {
    'n_estimators': 800,
    'learning_rate': 0.01,
    'max_depth': 4,
    'min_child_weight': 10,
    'subsample': 0.6,
    'colsample_bytree': 0.8,
    'reg_lambda': 5,
    'reg_alpha': 2,
    'random_state': 42,
    'n_jobs': -1
}

median_params = base_params.copy()
median_params.update({
    'max_depth': 8,            # DEEPER
    'min_child_weight': 5,     # Slightly more flexible
})

# --- Train lower quantile model (e.g., 10th percentile) ---
model_lower = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.025, **base_params)
model_lower.fit(X_train_scaled, y_train_np.ravel())

# --- Train upper quantile model (e.g., 90th percentile) ---
model_upper = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.975, **base_params)
model_upper.fit(X_train_scaled, y_train_np.ravel())

# --- Train median model (50th percentile) ---
model_median = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, **median_params)
model_median.fit(X_train_scaled, y_train_np.ravel())

# --- Save models ---
joblib.dump(model_lower, os.path.join(output_path, f"xgb_model_lower_{sampleSize}x{loci}.joblib"))
joblib.dump(model_upper, os.path.join(output_path, f"xgb_model_upper_{sampleSize}x{loci}.joblib"))
joblib.dump(model_median, os.path.join(output_path, f"xgb_model_median_{sampleSize}x{loci}.joblib"))

# --- Predict on input population Z ---
Z_scaled = scaler.transform(Z)
lower_bound = model_lower.predict(Z_scaled)
upper_bound = model_upper.predict(Z_scaled)
median_pred = model_median.predict(Z_scaled)

# --- Report (assuming Z has a single row) ---
print("\nXGBoost Quantile Regression Prediction on Z (Single Input)")
print(f"Median: {median_pred.item():.4f}, 95% CI: [{lower_bound.item():.4f}, {upper_bound.item():.4f}]")

# --- Evaluate model_median on test set ---
y_pred_median = model_median.predict(X_test_scaled)

mse_xgb = mean_squared_error(y_test_np, y_pred_median)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test_np, y_pred_median)
r2_xgb = r2_score(y_test_np, y_pred_median)

print("\nMedian Model (Quantile Regression) Metrics on Test Set")
print(f"MSE: {mse_xgb:.2f}, RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}, R2: {r2_xgb:.2f}")

# --- Evaluate cross-validation performance ---
evaluate_cv(model_median, X_train_scaled, y_train_np.ravel())


# Feature Importances
importances = model_median.feature_importances_
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

evaluate_cv(ridge_model, X_train_scaled, y_train_np.ravel())

print("\nRidge Feature Importances:")
for name, coef in sorted(zip(feature_names, ridge_coef), key=lambda x: abs(x[1]), reverse=True):
    print(f"{name}: {coef:.4f}")

print(f"\nLasso => RMSE: {lasso_rmse:.4f}, MAE: {lasso_mae:.4f}, R2: {lasso_r2:.4f}")
print_stats_inline("Lasso", lasso_stats)

evaluate_cv(lasso_model, X_train_scaled, y_train_np.ravel())

print("\nLasso Feature Importances:")
for name, coef in sorted(zip(feature_names, lasso_coef), key=lambda x: abs(x[1]), reverse=True):
    print(f"{name}: {coef:.4f}")

print("")
print("----- %s seconds -----" % (time.time() - start_time))

'''
