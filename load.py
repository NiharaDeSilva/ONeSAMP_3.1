from joblib import load
from statistics import statisticsClass
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

parser = argparse.ArgumentParser()
parser.add_argument("--o", type=str, help="The File Name")
#parser.add_argument("--m", type=str, help="Model Name")

args = parser.parse_args()

inputFileStatistics = statisticsClass()

fileName = "oneSampIn"
if (args.o):
    fileName = str(args.o)
else:
    print("WARNING:main: No filename provided.  Using oneSampIn")
'''
if (args.m):
    modelName = str(args.m)
else:
    print("WARNING:main: No filename provided.  Using oneSampIn")
'''
# t = time.time()
inputFileStatistics.readData(fileName)
inputFileStatistics.filterIndividuals(0.2)
inputFileStatistics.filterLoci(0.2)
#if (args.n):
#    inputFileStatistics.filterMonomorphicLoci()
inputFileStatistics.test_stat1()
inputFileStatistics.test_stat2()
inputFileStatistics.test_stat3()
inputFileStatistics.test_stat5()
inputFileStatistics.test_stat4()

inputStatsList = [str(inputFileStatistics.stat1), str(inputFileStatistics.stat2), str(inputFileStatistics.stat3),
             str(inputFileStatistics.stat4), str(inputFileStatistics.stat5)]

inputStatsList = pd.DataFrame([inputStatsList], columns = ['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt'])
Z = np.array(inputStatsList[['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']])

# -------------------------------
# Random Forest Regression
# -------------------------------

loci = inputFileStatistics.numLoci
sampleSize = inputFileStatistics.sampleSize
output_path = (f"/blue/boucher/suhashidesilva/2025/ONeSAMP_3/output_200_1/genePop{sampleSize}x{loci}")
os.makedirs(output_path, exist_ok=True)

print(f"\n-----------------RANDOM FOREST------------")

rf_path = os.path.join(output_path, f"rf_model_{sampleSize}x{loci}.joblib")
scalar_path = os.path.join(output_path, f"scaler_{sampleSize}x{loci}.joblib")

# --- Load the model and scaler (for future use) ---
rf_model = joblib.load(os.path.join(rf_path)
scaler = joblib.load(os.path.join(scalar_path)
Z_scaled = scaler.transform(Z)

# --- Make predictions ---
rf_prediction = rf_model.predict(Z_scaled)

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

# -------------------------------
# XGBoost
# -------------------------------

print(f"\n-----------------XGBoost------------------")

# --- Save the trained model and scaler ---
xgb_path = os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib")

# Load model and scaler
xgb_model = joblib.load(os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib"))

# Predict again using new Z
Z_scaled = scaler.transform(Z)
xgb_prediction = xgb_model.predict(Z_scaled)

# --- Predict on test set and input population Z ---
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

print("")

# -------------------------------
# Lasso & Ridge Regression
# -------------------------------

print(f"\n-----------------Lasso & Ridge Regression------------------")

ridge_path = os.path.join(output_path, f"ridge_model_{sampleSize}x{loci}.joblib")
lasso_path = os.path.join(output_path, f"lasso_model_{sampleSize}x{loci}.joblib")

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

ridge_loaded = joblib.load(os.path.join(ridge_path)
lasso_loaded = joblib.load(os.path.join(lass_path)

ridge_stats = predict_with_stats(ridge_loaded, X_train_scaled, y_train_np, Z_scaled)
lasso_stats = predict_with_stats(lasso_loaded, X_train_scaled, y_train_np, Z_scaled)

print_stats_inline("Ridge", ridge_stats)
print("")
print_stats_inline("Lasso", lasso_stats)

print("")
print("----- %s seconds -----" % (time.time() - start_time))
