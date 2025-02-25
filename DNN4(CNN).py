import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
data_path = r'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\partition_combined_data_upsampled_pm_3H_spline_1degreeM.csv'
confirmed = pd.read_csv(data_path)
confirmed['PM2.5N'] = confirmed['PM2.5'].shift(-1)
confirmed = confirmed[['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'Season', 'PM2.5', "PM2.5N"]].dropna()
print(f'Original Size: \n{confirmed}\n')
print(f'Dataframe: {confirmed}')

# Sequence creation functions
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

def prepare_data(confirmed, seq_length, train_size=0.7, val_size=0.15, batch_size=32):
    X = confirmed[['TP', 'WS', 'AP', 'HM', 'WD', 'PCPT', 'Season', 'PM2.5']].values.astype('float32')
    y = confirmed['PM2.5N'].values.astype('float32')
    X, y = create_sequences(X, y, seq_length)

    train_idx = int(len(X) * train_size)
    val_idx = int(len(X) * (train_size + val_size))

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    # Min-max scaling
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
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return (train_loader, val_loader, test_loader), (min_val, max_val)

class PM25CNN(nn.Module):
    def __init__(self, n_features, cnn_layer, hidden_size, drop_out):
        super(PM25CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = n_features
        for i in range(cnn_layer):
            self.conv_layers.append(
                nn.Conv1d(in_channels=in_channels, out_channels=hidden_size, kernel_size=1, padding=1)
            )
            in_channels = hidden_size
            self.conv_layers.append(nn.ReLU())
        
        self.conv_layers.append(nn.Dropout(drop_out))
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_length, n_features) -> (batch, n_features, seq_length)
        x = x.transpose(1, 2)
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

# Updated hyperparameter space including batch_size
space = [
    Integer(3, 20, name='seq_length'),
    Integer(1, 5, name='cnn_layer'),
    Integer(16, 512, name='cnn_hidden'),
    Real(0.0, 0.5, name='cnn_dropout'),
    Real(1e-5, 1e-2, prior='log-uniform', name='lr'),
    Integer(300, 1000, name='epochs'),
    Integer(16, 256, name='batch_size')
]

@use_named_args(space)
def objective(**params):
    objective.trial_counter += 1
    seq_length = int(params['seq_length'])
    batch_size = int(params['batch_size'])
    
    (train_loader, val_loader, _), (min_val, max_val) = prepare_data(
        confirmed, seq_length, batch_size=batch_size
    )

    print(f'''
          
          seq_length: {seq_length},
          cnn_layer: {params['cnn_layer']},
          cnn_hidden: {params['cnn_hidden']},
          cnn_dropout: {params['cnn_dropout']},
          lr: {params['lr']},
          epochs: {params['epochs']},
          batch_size: {batch_size}
          
          ''')
    print('---' * 10 + ' Start Model ' + '---' * 10)

    model = PM25CNN(
        n_features=8,
        cnn_layer=int(params['cnn_layer']),
        hidden_size=int(params['cnn_hidden']),
        drop_out=float(params['cnn_dropout'])
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=0.00001)
    loss_fn = nn.MSELoss()

    # Training loop with batches
    model.train()
    for epoch in range(params['epochs']):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_fn(y_pred.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        
        if (epoch + 1) % 100 == 0 or epoch == params['epochs'] - 1:
            avg_loss = epoch_loss / len(train_loader.dataset)
            print(f'Epoch [{epoch + 1}/{params["epochs"]}], Loss: {avg_loss:.6f}')

    print('---' * 10 + ' Start Model Evaluate ' + '---' * 10)

    # Validation with batches
    model.eval()
    val_loss = 0
    predictions = []
    true_vals = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            y_val_pred = model(batch_X)
            loss = loss_fn(y_val_pred.squeeze(), batch_y)
            val_loss += loss.item() * batch_X.size(0)
            predictions.extend(y_val_pred.squeeze().cpu().numpy())
            true_vals.extend(batch_y.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f'Average validation loss: {avg_val_loss}')

    # Rescale predictions
    pred = np.array(predictions)
    true_vals = np.array(true_vals)
    pred_values = pred * (max_val - min_val) + min_val
    true_values = true_vals * (max_val - min_val) + min_val

    trial_data = {
        "evaluate loss": avg_val_loss,
        **params,
        'weight_decay': 0.00001,
        "trial_MAE": mean_absolute_error(true_values, pred_values),
        "trial_MSE": mean_squared_error(true_values, pred_values),
        "trial_RMSE": RMSE(true_values, pred_values),
        "trial_R2": r2_score(true_values, pred_values)
    }

    # Plot validation results
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", marker="o")
    plt.plot(pred_values, label="Predicted Values", marker="x")
    plt.xlabel("Index")
    plt.ylabel("PM2.5")
    plt.title("True vs Predicted (Validation Set)")
    plt.legend()
    plt.grid(True)
    plt.savefig(rf"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-pm25-result\trial_true_vs_pred_{objective.trial_counter}.png")
    plt.close()

    # Save trial results
    trial_filename = r"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-pm25-result\trial_hyperparameters.csv"
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
(train_loader, val_loader, test_loader), (min_val, max_val) = prepare_data(
    confirmed, 
    best_params['seq_length'],
    batch_size=best_params['batch_size']
)

final_model = PM25CNN(
    n_features=8,
    cnn_layer=int(best_params['cnn_layer']),
    hidden_size=int(best_params['cnn_hidden']),
    drop_out=best_params['cnn_dropout']
).to(device)

optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], weight_decay=0.00001)
loss_fn = nn.MSELoss()

print("\nTraining final model with best parameters...")
final_model.train()
for epoch in range(best_params['epochs']):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        y_pred = final_model(batch_X)
        loss = loss_fn(y_pred.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    
    if (epoch + 1) % 100 == 0:
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{best_params["epochs"]}], Loss: {avg_loss:.6f}')

# Evaluation on Training and Test sets
final_model.eval()
train_preds = []
train_true = []
with torch.no_grad():
    for batch_X, batch_y in train_loader:
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

# Save and plot results
results_df = pd.DataFrame({
    'True Values': test_true_inv,
    'Predicted Values': test_preds_inv
})

metrics_df = pd.DataFrame({
    'weight_decay': 0.00001,
    'Train_MAE': [train_metrics['MAE']],
    'Train_MSE': [train_metrics['MSE']],
    'Train_RMSE': [train_metrics['RMSE']],
    'Train_R2': [train_metrics['R2']],
    'Test_MAE': [test_metrics['MAE']],
    'Test_MSE': [test_metrics['MSE']],
    'Test_RMSE': [test_metrics['RMSE']],
    'Test_R2': [test_metrics['R2']],
    'Best_Sequence_Length': [best_params['seq_length']],
    'Best_Batch_Size': [best_params['batch_size']]  # Added batch size to metrics
})

optimization_results = pd.DataFrame({
    'Trial': range(len(result.func_vals)),
    'Loss': result.func_vals
})
optimization_results.to_csv(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-pm25-results_optimization.csv', index=False)

results_df.to_csv(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-pm25-results_predictions.csv', index=False)
metrics_df.to_csv(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-pm25-results_metrics.csv', index=False)

# Plotting training and test results
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(train_true_inv[:100], label='True Values', color='blue')
plt.plot(train_preds_inv[:100], label='Predicted Values', color='red')
plt.title('Training Set: First 100 Predictions vs True Values')
plt.xlabel('Time Step')
plt.ylabel('PM2.5')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_true_inv[:100], label='True Values', color='blue')
plt.plot(test_preds_inv[:100], label='Predicted Values', color='red')
plt.title('Test Set: First 100 Predictions vs True Values')
plt.xlabel('Time Step')
plt.ylabel('PM2.5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-pm25-result\final_predictions.png')
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
plt.savefig(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-pm25-result\optimization_progress.png')
plt.close()

# Save the model and related information
torch.save({
    'model_state_dict': final_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_params': best_params,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'scaling_params': {'min_val': min_val, 'max_val': max_val}
}, r'C:\Users\CO19\Downloads\CNN\best_model.pth')

print("\nResults, plots, and best model have been saved to the specified directories")

# Create and save optimization summary
optimization_summary = pd.DataFrame({
    'Parameter': list(best_params.keys()),
    'Best Value': list(best_params.values())
})
optimization_summary.to_csv(rf'D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\CNN-pm25-result\best_params.csv', index=False)

print("\nOptimization summary:")
print(optimization_summary)
print("\nTraining complete!")