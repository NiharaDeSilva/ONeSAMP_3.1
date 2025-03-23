import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch.nn.init as init

'''
Feed Forward Neural Network
'''

def init_weights(m):
    if isinstance(m, nn.Linear):  # Apply only to Linear layers
        init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming He initialization
        if m.bias is not None:
            init.zeros_(m.bias)  # Set bias to zero


class FeedForwardNN(nn.Module):

    def __init__(self, input_size=5):
        super(FeedForwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 125),
            nn.ReLU(),
            nn.Linear(125, 250),
            nn.ReLU(),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.Linear(500, 125),
            nn.ReLU(),
            nn.Linear(125, 25),
            nn.ReLU(),
            nn.Linear(25, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        self.apply(init_weights)



    def forward(self, x):
        return self.model(x)

class PopulationGeneticsModel:
    def __init__(self, learning_rate=0.00005, epochs=100, batch_size=32):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = FeedForwardNN().to(self.device)
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
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    X_batch = X_train[start:start+self.batch_size]
                    y_batch = y_train[start:start+self.batch_size]
                    y_pred = self.model(X_batch)
                    loss = self.loss_fn(y_pred, y_batch)

                    self.optimizer.zero_grad()
                    loss.backward()

                    self.optimizer.step()
                    bar.set_postfix(mse=float(loss))

            self.model.eval()
            y_pred = self.model(X_test)
            mse = float(self.loss_fn(y_pred, y_test))
            self.history.append(mse)
            if mse < self.best_mse:
                self.best_mse = mse
                self.best_weights = copy.deepcopy(self.model.state_dict())
            print(f"Epoch {epoch+1}: Training Loss = {loss.item()}")


        self.model.load_state_dict(self.best_weights)

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test).cpu().numpy()
            y_test = y_test.cpu().numpy()

        #y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        #y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))
        
        absolute_errors = np.abs(y_pred - y_test)
        squared_errors = (y_pred - y_test) ** 2
        for i in range(min(10, len(squared_errors))):  # Print only first 5 rows
            print(f"Row {i+1}: True={y_test[i]}, Pred={y_pred[i]}, Error²={squared_errors[i]}")

        mse = np.mean(squared_errors) 
        rmse = np.sqrt(mse)

        stats = {
            'MAE': round(np.mean(absolute_errors), 2),
            'MSE' : round(mse, 2) ,
            'RMSE' : round(rmse, 2)
        }
        return stats

    def predict_with_uncertainty(self, Z, n_simulations=100, noise_std_factor=0.01):
        self.model.eval()
        predictions = np.zeros(n_simulations)
        noise_std = noise_std_factor * torch.std(Z)

        for i in range(n_simulations):
            random = torch.randn(Z.shape).to(self.device)
            Z_perturbed = Z + (random * noise_std)
            with torch.no_grad():
                pred = self.model(Z_perturbed).cpu().numpy().flatten()[0]
                predictions[i] = pred
        return [
            round(np.min(predictions), 2),
            round(np.max(predictions), 2),
            round(np.mean(predictions), 2),
            round(np.median(predictions), 2),
            round(np.percentile(predictions, 2.5) ,2), round(np.percentile(predictions, 97.5),2)
        ]

















'''
# Convert to PyTorch tensors
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# train-test split for model evaluation
# X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Standardizing data
# scaler = StandardScaler()
# scaler.fit(X_train_raw)
# X_train = scaler.transform(X_train_raw)
# X_test = scaler.transform(X_test_raw)


X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = y_train.astype(np.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
y_test = y_test.astype(np.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)
Z = Z.astype(np.float32)
Z = torch.tensor(Z, dtype=torch.float32).to(device)

# # Convert to 2D PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(5, 100),
    nn.ReLU(),
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
model.to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

print(f"\n-----------------NEURAL NETWORK------------------")

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))

# Convert tensors to numpy arrays
y_test = y_test.numpy()
y_pred = y_pred.detach().numpy()  # Ensure y_pred is detached from the computation graph

# Calculate absolute errors
absolute_errors = np.abs(y_pred - y_test)
min = np.min(absolute_errors)
max = np.max(absolute_errors)
q1 = np.percentile(absolute_errors, 25)
median = np.percentile(absolute_errors, 50)
q3 = np.percentile(absolute_errors, 75)
mae = np.mean(absolute_errors)
print(f"MAE: {mae:.2f}")
print(f"{min:.2f} {max:.2f} {median:.2f} {q1:.2f} {q3:.2f}")
# ##########################
#plt.plot(history)
#plt.show()

model.eval()

# Number of simulations
n_simulations = 100
# Array to store predictions
predictions = np.zeros(n_simulations)

# Standard deviation of noise to add to Z for simulations
# Adjust the scale based on your expected input variability
noise_std = 0.01 * torch.std(Z)

for i in range(n_simulations):
    # Add random noise to Z
    random = torch.randn(Z.shape)
    Z_perturbed = Z + (random * noise_std)
    # Predict with model
    with torch.no_grad():
        pred = model(Z_perturbed)
    # Ensure pred is converted to a scalar if necessary, assuming pred should be a single value
    pred_scalar = pred.numpy().flatten()[0]  # Flatten and take the first element to ensure scalar conversion
    predictions[i] = pred_scalar

# Calculate confidence interval
lower = np.percentile(predictions, 2.5)
upper = np.percentile(predictions, 97.5)

print(f"Neural network prediction: ")
print(f"{np.mean(predictions).round(2)}")
print(f"median prediction: {np.median(predictions).round(2)}")
print(f"95% confidence interval:")
print(f"{lower:.2f}, {upper:.2f}")

print("----- %s seconds -----" % (time.time() - start_time))
'''
