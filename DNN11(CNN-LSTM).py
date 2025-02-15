import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Metric functions
def MAE(true, pred):
    return np.mean(np.abs(true - pred))
def MSE(true, pred):
    return np.mean((true - pred) ** 2)
def RMSE(true, pred):
    return np.sqrt(MSE(true, pred))
def R2(true, pred):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - ss_res / ss_tot

# Plotting setup
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = (14, 10)
register_matplotlib_converters()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Data loading
data_path = rf'Y:\รันDeepในเครื่องอาจารย์บุ๊ค\partition_combined_data_upsampled_pm_3H_spline_1degreeM.csv'
confirmed = pd.read_csv(data_path)
confirmed = confirmed[['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'PM2.5', 'Season']]
print(f'Original Size: \n{confirmed}\n')
confirmed['PM2.5B'] = confirmed['PM2.5'].shift(1)
confirmed = confirmed[['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'PM2.5', 'Season', 'PM2.5B']]
print(f'Describe Dataframe: \n{confirmed.describe().to_string()}')
confirmed = confirmed[['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'PM2.5', 'Season', 'PM2.5B']].values.astype('float32')
print(f'Dataframe: {confirmed}')

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length, 0])
    return np.array(xs), np.array(ys)

def prepare_data(data, seq_length, train_size=0.7, val_size=0.15):
    X, y = create_sequences(data, seq_length)
    
    train_idx = int(len(X) * train_size)
    val_idx = int(len(X) * (train_size + val_size))
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    min_val = X_train[:, :, 0].min()
    max_val = X_train[:, :, 0].max()
    
    def MinMaxScale(array, min_val, max_val):
        return (array - min_val) / (max_val - min_val)
    
    X_train = MinMaxScale(X_train, min_val, max_val)
    y_train = MinMaxScale(y_train, min_val, max_val)
    X_val = MinMaxScale(X_val, min_val, max_val)
    y_val = MinMaxScale(y_val, min_val, max_val)
    X_test = MinMaxScale(X_test, min_val, max_val)
    y_test = MinMaxScale(y_test, min_val, max_val)
    
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test), (min_val, max_val)

class Model(nn.Module):
    def __init__(self, n_features, cnn_hidden, cnn_dropout,
                 lstm_hidden, lstm_layers, lstm_dropout, seq_len):
        super(Model, self).__init__()
        self.seq_len = seq_len
        fixed_kernel = 2
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=cnn_hidden,
                      kernel_size=fixed_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=fixed_kernel, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(cnn_dropout)
        )
        self.lstm = nn.LSTM(input_size=cnn_hidden, hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
                            batch_first=True)
        self.linear = nn.Linear(lstm_hidden, 1)

    def forward(self, sequences):
        x = sequences.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        y_pred = self.linear(last_time_step)
        return y_pred

# Define the optimization space
space = [
    Integer(3, 20, name='seq_length'),
    Integer(16, 64, name='cnn_hidden'),
    Real(0.0, 0.5, name='cnn_dropout'),
    Integer(16, 64, name='lstm_hidden'),
    Integer(1, 5, name='lstm_layers'),
    Real(0.0, 0.5, name='lstm_dropout'),
    Real(1e-5, 1e-2, prior='log-uniform', name='lr'),
    Integer(1000, 1500, name='epochs')
]

# Objective function for Bayesian optimization
@use_named_args(space)
def objective(**params):
    seq_length = params['seq_length']
    (X_train, y_train, X_val, y_val, _, _), (min_val, max_val) = prepare_data(confirmed, seq_length)
    
    print(f'''
          seq_length : {seq_length},
          cnn_hidden : {params['cnn_hidden']},
          cnn_dropout : {params['cnn_dropout']},
          lstm_hidden : {params['lstm_hidden']},
          lstm_layers : {params['lstm_layers']},
          lstm_dropout  : {params['lstm_dropout']},
          lr : {params['lr']},
          epochs : {params['epochs']}
          ''')
    
    print(f'---'*5 + ' Start Model ' + '---'*5)
    
    model = Model(
        n_features=8,
        cnn_hidden=params['cnn_hidden'],
        cnn_dropout=params['cnn_dropout'],
        lstm_hidden=params['lstm_hidden'],
        lstm_layers=params['lstm_layers'],
        lstm_dropout=params['lstm_dropout'],
        seq_len=seq_length
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    loss_fn = nn.MSELoss()
    
    # Training
    model.train()
    for epoch in range(params['epochs']):
        for idx, seq in enumerate(X_train):
            optimizer.zero_grad()
            seq = torch.unsqueeze(seq, 0)
            y_pred = model(seq)
            loss = loss_fn(y_pred.squeeze(), y_train[idx])
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == params['epochs'] - 1:
            print(f'Epoch [{epoch + 1}/{params["epochs"]}]')
    
    print(f'---'*5 + 'Start Model Evaluate' + '---'*5)
    
    # Validation
    model.eval()
    val_loss = 0
    predictions = []
    true_vals = []
    with torch.no_grad():
        for idx, seq in enumerate(X_val):
            seq = torch.unsqueeze(seq, 0)
            y_val_pred = model(seq)
            loss = loss_fn(y_val_pred.squeeze(), y_val[idx])
            val_loss += loss.item()
            predictions.append(y_val_pred.squeeze().cpu().numpy())
            true_vals.append(y_val[idx].cpu().numpy())
    
    avg_val_loss = val_loss / len(X_val)
    print(f'Average validation loss: {avg_val_loss}')
    
    # Save trial results
    trial_data = {
        "value": avg_val_loss,
        **params
    }
    
    pred = np.array(predictions)
    true_vals = np.array(true_vals)
    pred_values = pred * (max_val - min_val) + min_val
    true_values = true_vals * (max_val - min_val) + min_val
    
    trial_data.update({
        "trial_MAE": mean_absolute_error(true_values, pred_values),
        "trial_MSE": mean_squared_error(true_values, pred_values),
        "trial_RMSE": RMSE(true_values, pred_values),
        "trial_R2": r2_score(true_values, pred_values)
    })
    
    # Plot and save validation results
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", marker="o")
    plt.plot(pred_values, label="Predicted Values", marker="x")
    plt.xlabel("Index")
    plt.ylabel("Air Quality Index")
    plt.title(f"True vs Predicted (Validation Set)")
    plt.legend()
    plt.grid(True)
    plt.savefig(rf"Y:\รันDeepในเครื่องอาจารย์บุ๊ค\CNN-LSTM\fig\trial_true_vs_pred_{len(pred_values)}.png")
    plt.close()
    
    # Save trial data
    trial_filename = rf"Y:\รันDeepในเครื่องอาจารย์บุ๊ค\CNN-LSTM\result\trial_hyperparameters.csv"
    try:
        df_existing = pd.read_csv(trial_filename)
        df_new = pd.DataFrame([trial_data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_combined = pd.DataFrame([trial_data])
    
    df_combined.to_csv(trial_filename, index=False)
    
    return avg_val_loss

# Run Bayesian optimization
result = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=50,
    n_random_starts=10,
    noise=0.1,
    random_state=RANDOM_SEED
)

# Get best parameters
best_params = dict(zip([dim.name for dim in space], result.x))
print("\nBest parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Train final model with best parameters
best_seq_length = best_params['seq_length']
(X_train, y_train, X_val, y_val, X_test, y_test), (min_val, max_val) = prepare_data(confirmed, best_seq_length)

final_model = Model(
    n_features=8,
    cnn_hidden=best_params['cnn_hidden'],
    cnn_dropout=best_params['cnn_dropout'],
    lstm_hidden=best_params['lstm_hidden'],
    lstm_layers=best_params['lstm_layers'],
    lstm_dropout=best_params['lstm_dropout'],
    seq_len=best_seq_length
).to(device)

optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
loss_fn = nn.MSELoss()

# Train final model
print("\nTraining final model with best parameters...")
final_model.train()
for epoch in range(best_params['epochs']):
    for idx, seq in enumerate(X_train):
        optimizer.zero_grad()
        seq = torch.unsqueeze(seq, 0)
        y_pred = final_model(seq)
        loss = loss_fn(y_pred.squeeze(), y_train[idx])
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{best_params["epochs"]}]')

# Evaluate on train set
final_model.eval()
train_preds = []
with torch.no_grad():
    for seq in X_train:
        seq = torch.unsqueeze(seq, 0)
        pred = final_model(seq)
        train_preds.append(pred.squeeze().cpu().item())

train_true = y_train.cpu().numpy()
train_preds = np.array(train_preds)
train_preds_inv = train_preds * (max_val - min_val) + min_val
train_true_inv = train_true * (max_val - min_val) + min_val

# Evaluate on test set
test_preds = []
with torch.no_grad():
    for seq in X_test:
        seq = torch.unsqueeze(seq, 0)
        pred = final_model(seq)
        test_preds.append(pred.squeeze().cpu().item())

test_true = y_test.cpu().numpy()
test_preds = np.array(test_preds)
test_preds_inv = test_preds * (max_val - min_val) + min_val
test_true_inv = test_true * (max_val - min_val) + min_val

# Calculate metrics
train_metrics = {
    'MAE': mean_absolute_error(train_true_inv, train_preds_inv),
    'MSE': mean_squared_error(train_true_inv, train_preds_inv),
    'RMSE': np.sqrt(mean_squared_error(train_true_inv, train_preds_inv)),
    'R2': r2_score(train_true_inv, train_preds_inv)
}

test_metrics = {
    'MAE': mean_absolute_error(test_true_inv, test_preds_inv),
    'MSE': mean_squared_error(test_true_inv, test_preds_inv),
    'RMSE': np.sqrt(mean_squared_error(test_true_inv, test_preds_inv)),
    'R2': r2_score(test_true_inv, test_preds_inv)
}

# Print metrics
print("\nTraining Metrics:")
for metric, value in train_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nTest Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save results
results_df = pd.DataFrame({
    'True Values': test_true_inv,
    'Predicted Values': test_preds_inv
})

metrics_df = pd.DataFrame({
    'Train_MAE': [train_metrics['MAE']],
    'Train_MSE': [train_metrics['MSE']],
    'Train_RMSE': [train_metrics['RMSE']],
    'Train_R2': [train_metrics['R2']],
    'Test_MAE': [test_metrics['MAE']],
    'Test_MSE': [test_metrics['MSE']],
    'Test_RMSE': [test_metrics['RMSE']],
    'Test_R2': [test_metrics['R2']],
    'Best_Sequence_Length': [best_seq_length]
})

# Save optimization results
optimization_results = pd.DataFrame({
    'Trial': range(len(result.func_vals)),
    'Loss': result.func_vals
})
optimization_results.to_csv(rf'Y:\รันDeepในเครื่องอาจารย์บุ๊ค\CNN-LSTM\results(CNN-LSTM)_optimization.csv', index=False)

# Save results to CSV
results_df.to_csv(rf'Y:\รันDeepในเครื่องอาจารย์บุ๊ค\CNN-LSTM\results(CNN-LSTM)_predictions.csv', index=False)
metrics_df.to_csv(rf'Y:\รันDeepในเครื่องอาจารย์บุ๊ค\CNN-LSTM\results(CNN-LSTM)_metrics.csv', index=False)

# Plot final results
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(train_true_inv[:100], label='True Values', color='blue')
plt.plot(train_preds_inv[:100], label='Predicted Values', color='red')
plt.title('Training Set: First 100 Predictions vs True Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_true_inv[:100], label='True Values', color='blue')
plt.plot(test_preds_inv[:100], label='Predicted Values', color='red')
plt.title('Test Set: First 100 Predictions vs True Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(rf'Y:\รันDeepในเครื่องอาจารย์บุ๊ค\CNN-LSTM\fig\final_predictions.png')
plt.close()

# Plot optimization progress
plt.figure(figsize=(10, 6))
plt.plot(result.func_vals, 'b-', label='Objective value')
plt.plot(result.func_vals.cummin(), 'r-', label='Best value')
plt.xlabel('Number of calls')
plt.ylabel('Objective value')
plt.title('Bayesian Optimization Progress')
plt.legend()
plt.grid(True)
plt.savefig(rf'Y:\รันDeepในเครื่องอาจารย์บุ๊ค\CNN-LSTM\fig\optimization_progress.png')
plt.close()

# Save the best model
torch.save({
    'model_state_dict': final_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_params': best_params,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'scaling_params': {'min_val': min_val, 'max_val': max_val}
}, rf'Y:\รันDeepในเครื่องอาจารย์บุ๊ค\CNN-LSTM\best_model.pth')

print("\nResults, plots, and best model have been saved to the specified directories")

# Save hyperparameter optimization summary
optimization_summary = pd.DataFrame({
    'Parameter': list(best_params.keys()),
    'Best Value': list(best_params.values())
})
optimization_summary.to_csv(rf'Y:\รันDeepในเครื่องอาจารย์บุ๊ค\CNN-LSTM\results(CNN-LSTM)_best_params.csv', index=False)

print("\nOptimization summary:")
print(optimization_summary)
print("\nTraining complete!")