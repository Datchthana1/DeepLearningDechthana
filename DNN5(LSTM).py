import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os
from pprint import pprint
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class PollutionPredictor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self, file_path):
        df = pd.read_csv(file_path)

        if 'date' in df.columns:
            df = df.drop(columns='date')

        if 'wnd_dir' in df.columns:
            df["wnd_dir"] = self.label_encoder.fit_transform(df['wnd_dir'])

        data = df.values
        return data, df.columns

    def preprocess_data(self, data):
        scaled_data = self.scaler.fit_transform(data)
        return torch.tensor(scaled_data, dtype=torch.float32)

    def create_sequences(self, data, input_window_size):
        sequences = []
        for i in range(len(data) - input_window_size):
            seq = data[i:i + input_window_size]
            label = data[i + input_window_size, 0]
            sequences.append((seq, label))
        return sequences

    def split_data(self, data_tensor, test_size=0.2):
        total_sequences = self.create_sequences(data_tensor, self.config['input_window_size'])
        train_sequences, val_sequences = train_test_split(total_sequences, test_size=test_size)
        return train_sequences, val_sequences

    def objective(self, trial):
        config = {
            'input_window_size': trial.suggest_int('input_window_size', 5, 20),
            'hidden_size': trial.suggest_int('hidden_size', 16, 256, step=8),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'epochs': 100,
            'patience': 30
        }
        self.config = config

        train_data, _ = self.load_data(
            r"D:\DataCenter\รันDeepในเครื่องอาจารย์บุ๊ค\archive (14)\LSTM-Multivariate_pollution.csv")
        train_data_tensor = self.preprocess_data(train_data)
        train_sequences, val_sequences = self.split_data(train_data_tensor)

        model = LSTMModel(
            input_size=train_data.shape[1],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(config['epochs']):
            model.train()
            train_epoch_loss = self._run_epoch(model, train_sequences, criterion, optimizer, is_training=True)

            model.eval()
            val_epoch_loss = self._run_epoch(model, val_sequences, criterion, optimizer, is_training=False)

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config['patience']:
                break

        return best_val_loss

    def _run_epoch(self, model, sequences, criterion, optimizer, is_training=True):
        total_loss = 0
        for seq, label in sequences:
            seq, label = seq.to(self.device), label.to(self.device)

            if is_training:
                optimizer.zero_grad()

            output = model(seq.unsqueeze(0))
            loss = criterion(output, label.unsqueeze(0).unsqueeze(0))

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(sequences)

    def _plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('loss_curve.png')
        plt.close()

    def _plot_predictions(self, true_values, predicted_values):
        plt.figure(figsize=(12, 6))
        plt.plot(true_values, label='True Values', color='blue')
        plt.plot(predicted_values, label='Predicted Values', color='red', linestyle='--')
        plt.legend()
        plt.title('True Values vs Predicted Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Pollution Levels')
        plt.savefig('predicted_values.png')
        plt.close()

    def train(self, train_sequences, val_sequences, train_data):
        model = LSTMModel(
            input_size=train_data.shape[1],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers']
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            model.train()
            train_epoch_loss = self._run_epoch(model, train_sequences, criterion, optimizer, is_training=True)

            model.eval()
            val_epoch_loss = self._run_epoch(model, val_sequences, criterion, optimizer, is_training=False)

            train_losses.append(train_epoch_loss)
            val_losses.append(val_epoch_loss)

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

            if (epoch + 1) % 20 == 0:
                print(
                    f'Epoch [{epoch + 1}/{self.config["epochs"]}], Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

        self._plot_losses(train_losses, val_losses)

        return model

    def evaluate(self, model, test_sequences):
        model.eval()
        predicted_values, true_values = []

        with torch.no_grad():
            for seq, label in test_sequences:
                seq, label = seq.to(self.device), label.to(self.device)
                output = model(seq.unsqueeze(0))
                predicted_values.append(output.item())
                true_values.append(label.item())

        predicted_values = self.scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
        true_values = self.scaler.inverse_transform(np.array(true_values).reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
        mae = mean_absolute_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)

        print(f'RMSE: {rmse}')
        print(f'MAE: {mae}')
        print(f'R2 Score: {r2}')

        self._plot_predictions(true_values, predicted_values)

        return predicted_values, true_values


def main():
    os.makedirs('output', exist_ok=True)
    os.chdir('output')

    study = optuna.create_study(direction='minimize')
    predictor = PollutionPredictor({})
    study.optimize(predictor.objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (validation loss): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_config = {**trial.params, 'epochs': 200, 'patience': 50}
    predictor = PollutionPredictor(best_config)

    train_data, _ = predictor.load_data(
        r"D:\DataCenter\รันDeepในเครื่องอาจารย์บุ๊ค\archive (14)\LSTM-Multivariate_pollution.csv")
    train_data_tensor = predictor.preprocess_data(train_data)
    train_sequences, val_sequences = predictor.split_data(train_data_tensor)

    model = predictor.train(train_sequences, val_sequences, train_data)

    test_data, _ = predictor.load_data(r"D:\DataCenter\รันDeepในเครื่องอาจารย์บุ๊ค\archive (14)\pollution_test_data1.csv")
    test_data_tensor = predictor.preprocess_data(test_data)

    test_sequences = predictor.create_sequences(test_data_tensor, best_config['input_window_size'])

    predictor.evaluate(model, test_sequences)


if __name__ == "__main__":
    main()
