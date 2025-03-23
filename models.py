# models.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import tqdm
import numpy as np
import copy
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


class FeedForwardNN(nn.Module):
    def __init__(self, input_size=5):
        super(FeedForwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.Linear(100, 25),
            nn.ReLU(),
            nn.Linear(25, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        self.apply(self.init_weights)

    def forward(self, x):
        return self.model(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)


class PopulationGeneticsModel:
    def __init__(self, input_size=5, learning_rate=0.0005, epochs=100, batch_size=32):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = FeedForwardNN(input_size).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.best_weights = None
        self.best_mse = np.inf
        self.history = []

    def train(self, X_train, y_train, X_test, y_test):
        batch_start = torch.arange(0, len(X_train), self.batch_size)
        for epoch in range(self.epochs):
            self.model.train()
            for start in batch_start:
                X_batch = X_train[start:start+self.batch_size]
                y_batch = y_train[start:start+self.batch_size]
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            y_pred = self.model(X_test)
            mse = float(self.loss_fn(y_pred, y_test))
            self.history.append(mse)
            if mse < self.best_mse:
                self.best_mse = mse
                self.best_weights = copy.deepcopy(self.model.state_dict())
            print(f"Epoch {epoch+1}: Training Loss = {loss.item():.4f}")

        self.model.load_state_dict(self.best_weights)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X).cpu().numpy().flatten()

    def predict_with_uncertainty(self, X, n_simulations=100):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(n_simulations):
                preds = self.model(X).cpu().numpy()
                predictions.append(preds)
        return {
            "mean": np.mean(predictions, axis=0),
            "std": np.std(predictions, axis=0)
        }

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test).cpu().numpy()
            y_test = y_test.cpu().numpy()

        absolute_errors = np.abs(y_pred - y_test)
        squared_errors = (y_pred - y_test) ** 2
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)

        stats = {
            'MAE': round(np.mean(absolute_errors), 2),
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2)
        }
        return stats


def train_xgboost(X_train_np, y_train_np, X_test_np, y_test_np):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        tree_method="hist"
    )
    model.fit(X_train_np, y_train_np,
              eval_set=[(X_test_np, y_test_np)],
              early_stopping_rounds=20,
              verbose=False)
    return model


def ensemble_predict(nn_pred, xgb_pred, strategy='average'):
    if strategy == 'average':
        return (nn_pred + xgb_pred) / 2
    # Add other strategies if needed
    return nn_pred  # default fallback
