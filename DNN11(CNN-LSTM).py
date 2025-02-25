import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from torch.utils.data import DataLoader, TensorDataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Plotting setup
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = (14, 10)
register_matplotlib_converters()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Data loading
data_path = rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\partition_combined_data_upsampled_pm_3H_spline_1degreeM.csv'
confirmed = pd.read_csv(data_path)
confirmed['PM2.5N'] = confirmed['PM2.5'].shift(-1)
confirmed = confirmed[['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'PM2.5', 'Season', 'PM2.5N']].dropna()
print(f'Original Size: \n{confirmed}\n')
print(f'Dataframe: {confirmed}')

# Modified sequence creation functions to support batching
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

def prepare_data(confirmed, seq_length, train_size=0.7, val_size=0.15):
    X = confirmed[['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'Season', 'PM2.5']].values.astype('float32')
    y = confirmed['PM2.5N'].values.astype('float32')
    X, y = create_sequences(X, y, seq_length)

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

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Create DataLoader objects
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    return (train_dataset, val_dataset, test_dataset), (min_val, max_val)

class Model(nn.Module):
    def __init__(self, n_features, cnn_layers, cnn_hidden, cnn_dropout,
                 lstm_hidden, lstm_layers, lstm_dropout, seq_len):
        super(Model, self).__init__()
        self.seq_len = seq_len
        fixed_kernel = 2

        layers = []
        for i in range(cnn_layers):
            in_channels = n_features if i == 0 else cnn_hidden
            layers.append(nn.Conv1d(in_channels, cnn_hidden, kernel_size=fixed_kernel, stride=1, padding=1))
            layers.append(nn.BatchNorm1d(cnn_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=fixed_kernel, stride=1, padding=1))
        layers.append(nn.Dropout(cnn_dropout))
        self.cnn = nn.Sequential(*layers)

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

# Modified optimization space to include batch_size
space = [
    Integer(3, 20, name='seq_length'),
    Integer(1, 5, name='cnn_layers'),
    Integer(16, 512, name='cnn_hidden'),
    Real(0.0, 0.5, name='cnn_dropout'),
    Integer(16, 512, name='lstm_hidden'),
    Integer(1, 5, name='lstm_layers'),
    Real(0.0, 0.5, name='lstm_dropout'),
    Real(1e-5, 1e-2, prior='log-uniform', name='lr'),
    Integer(300, 1000, name='epochs'),
    Integer(16, 256, name='batch_size')  # Added batch_size parameter
]



@use_named_args(space)
def objective(**params):
    objective.trial_counter += 1
    seq_length = int(params['seq_length'])
    batch_size = int(params['batch_size'])
    (train_dataset, val_dataset, _), (min_val, max_val) = prepare_data(confirmed, seq_length)
    
    # Create DataLoaders with the current batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f'''
          seq_length : {seq_length},
          cnn_layers : {params['cnn_layers']},
          cnn_hidden : {params['cnn_hidden']},
          cnn_dropout : {params['cnn_dropout']},
          lstm_hidden : {params['lstm_hidden']},
          lstm_layers : {params['lstm_layers']},
          lstm_dropout : {params['lstm_dropout']},
          weight_decay: 0.00001,
          lr : {params['lr']},
          epochs : {params['epochs']},
          batch_size : {batch_size}
          ''')
    print('---' * 10 + ' Start Model ' + '---' * 10)

    model = Model(
        n_features=8,
        cnn_layers=int(params['cnn_layers']),
        cnn_hidden=int(params['cnn_hidden']),
        cnn_dropout=float(params['cnn_dropout']),
        lstm_hidden=int(params['lstm_hidden']),
        lstm_layers=int(params['lstm_layers']),
        lstm_dropout=float(params['lstm_dropout']),
        seq_len=int(seq_length)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0.00001)
    loss_fn = nn.MSELoss()

    # Modified training loop for batch processing
    model.train()
    for epoch in range(params['epochs']):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_fn(y_pred.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == params['epochs'] - 1:
            print(f'Epoch [{epoch + 1}/{params["epochs"]}]')

    print('---' * 10 + ' Start Model Evaluate ' + '---' * 10)

    model.eval()
    val_loss = 0
    predictions = []
    true_vals = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            y_val_pred = model(batch_X)
            loss = loss_fn(y_val_pred.squeeze(), batch_y)
            val_loss += loss.item() * batch_X.size(0)
            preds = y_val_pred.squeeze()
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
            predictions.extend(preds.cpu().numpy())
            true_vals.extend(batch_y.cpu().numpy())

    avg_val_loss = val_loss / len(val_dataset)
    print(f'Average validation loss: {avg_val_loss}')

    pred = np.array(predictions)
    true_vals = np.array(true_vals)
    pred_values = pred * (max_val - min_val) + min_val
    true_values = true_vals * (max_val - min_val) + min_val

    trial_data = {
        "value": avg_val_loss,
        **params,
        "trial_MAE": mean_absolute_error(true_values, pred_values),
        "trial_MSE": mean_squared_error(true_values, pred_values),
        "trial_RMSE": root_mean_squared_error(true_values, pred_values),
        "trial_R2": r2_score(true_values, pred_values)
    }

    # Plot validation results
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", marker="o")
    plt.plot(pred_values, label="Predicted Values", marker="x")
    plt.xlabel("Index")
    plt.ylabel("Air Quality Index")
    plt.title("True vs Predicted (Validation Set)")
    plt.legend()
    plt.grid(True)
    plt.savefig(rf"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-LSTM-PM2.5\fig\trial_true_vs_pred_{objective.trial_counter}.png")
    plt.close()

    # Save trial results
    trial_filename = r"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-LSTM-PM2.5\result\trial_hyperparameters.csv"
    try:
        df_existing = pd.read_csv(trial_filename)
        df_new = pd.DataFrame([trial_data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_combined = pd.DataFrame([trial_data])
    df_combined.to_csv(trial_filename, index=False)

    return avg_val_loss

objective.trial_counter = 0

# Run Bayesian optimization
result = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=100,
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
(train_dataset, val_dataset, test_dataset), (min_val, max_val) = prepare_data(confirmed, best_seq_length)

# Create DataLoaders with best batch size
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

final_model = Model(
    n_features=8,
    cnn_layers=best_params['cnn_layers'],
    cnn_hidden=best_params['cnn_hidden'],
    cnn_dropout=best_params['cnn_dropout'],
    lstm_hidden=best_params['lstm_hidden'],
    lstm_layers=best_params['lstm_layers'],
    lstm_dropout=best_params['lstm_dropout'],
    seq_len=best_seq_length
).to(device)

optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=0.00001)
loss_fn = nn.MSELoss()

print("\nTraining final model with best parameters...")
final_model.train()
for epoch in range(best_params['epochs']):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        y_pred = final_model(batch_X)
        loss = loss_fn(y_pred.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{best_params["epochs"]}]')

# Evaluation
final_model.eval()
train_preds = []
train_true = []
with torch.no_grad():
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        pred = final_model(batch_X)
        train_preds.extend(pred.squeeze().cpu().numpy())
        train_true.extend(batch_y.cpu().numpy())

train_preds = np.array(train_preds)
train_true = np.array(train_true)
train_preds_inv = train_preds * (max_val - min_val) + min_val
train_true_inv = train_true * (max_val - min_val) + min_val

test_preds = []
test_true = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        pred = final_model(batch_X)
        test_preds.extend(pred.squeeze().cpu().numpy())
        test_true.extend(batch_y.cpu().numpy())

test_preds = np.array(test_preds)
test_true = np.array(test_true)
test_preds_inv = test_preds * (max_val - min_val) + min_val
test_true_inv = test_true * (max_val - min_val) + min_val

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

print("\nTraining Metrics:")
for metric, value in train_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nTest Metrics:")
for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save results and create plots
results_df = pd.DataFrame({
    'True Values': test_true_inv,
    'Predicted Values': test_preds_inv
})

metrics_df = pd.DataFrame({
    'weigh_decay': 0.00001,
    'Train_MAE': [train_metrics['MAE']],
    'Train_MSE': [train_metrics['MSE']],
    'Train_RMSE': [train_metrics['RMSE']],
    'Train_R2': [train_metrics['R2']],
    'Test_MAE': [test_metrics['MAE']],
    'Test_MSE': [test_metrics['MSE']],
    'Test_RMSE': [test_metrics['RMSE']],
    'Test_R2': [test_metrics['R2']],
    'Best_Sequence_Length': [best_seq_length],
    'Best_Batch_Size': [best_params['batch_size']]  # Added batch size to metrics
})

optimization_results = pd.DataFrame({
    'Trial': range(len(result.func_vals)),
    'Loss': result.func_vals
})

# Save results to CSV files
optimization_results.to_csv(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-LSTM-PM2.5\results(CNN-LSTM)_optimization.csv', index=False)
results_df.to_csv(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-LSTM-PM2.5\results(CNN-LSTM)_predictions.csv', index=False)
metrics_df.to_csv(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-LSTM-PM2.5\results(CNN-LSTM)_metrics.csv', index=False)

# Create and save plots
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
plt.savefig(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-LSTM-PM2.5\fig\final_predictions.png')
plt.close()

# Plot optimization progress
plt.figure(figsize=(10, 6))
plt.plot(result.func_vals, 'b-', label='Objective value')
plt.plot(np.minimum.accumulate(result.func_vals), 'r-', label='Best value')
plt.xlabel('Number of calls')
plt.ylabel('Objective value')
plt.title('Bayesian Optimization Progress')
plt.legend()
plt.grid(True)
plt.savefig(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-LSTM-PM2.5\fig\optimization_progress.png')
plt.close()

# Save model and parameters
torch.save({
    'model_state_dict': final_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_params': best_params,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'scaling_params': {'min_val': min_val, 'max_val': max_val}
}, r'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-LSTM-PM2.5\best_model.pth')

# Create and save optimization summary
optimization_summary = pd.DataFrame({
    'Parameter': list(best_params.keys()),
    'Best Value': list(best_params.values())
})
optimization_summary.to_csv(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-LSTM-PM2.5\results(CNN-LSTM)_best_params.csv', index=False)

print("\nOptimization summary:")
print(optimization_summary)
print("\nTraining complete!")