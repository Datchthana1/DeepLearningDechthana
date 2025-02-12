import torch, optuna, logging, itertools, pandas as pd, matplotlib.pyplot as plt, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
This script trains a Deep Neural Network (DNN) model using PyTorch and Optuna for hyperparameter optimization.
It reads data from a CSV file, preprocesses it, and trains the model using various combinations of features.
The script also includes functionality for early stopping, logging, and plotting learning curves.
Modules:
    torch: PyTorch library for deep learning.
    optuna: Library for hyperparameter optimization.
    logging: Standard Python logging library.
    itertools: Standard Python library for creating iterators for efficient looping.
    pandas: Library for data manipulation and analysis.
    matplotlib: Library for creating static, animated, and interactive visualizations.
    sklearn: Library for machine learning, including model selection and evaluation.
Functions:
    plot_learning_curve(train_losses, test_accuracies, features_format):
        Plots the training loss and test accuracy curves.
Classes:
    DNN(nn.Module):
        Defines a Deep Neural Network model with configurable layers, activation functions, dropout, and batch normalization.
Usage:
    The script iterates over combinations of features, splits the data into training and testing sets, and uses Optuna to find the best hyperparameters.
    It then trains the model with the best hyperparameters and evaluates its performance on unseen data.
    The results, including the best hyperparameters and accuracies, are logged and saved to CSV files.
"""


# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] - %(levelname)s - %(message)s"
)

# เช็ค GPU
device = ("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# อ่านข้อมูล
df = pd.read_csv(
    rf"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\OverSamplingSMOTE.csv"
)

# นิยามคลาส DNN นอกฟังก์ชัน objective
class DNN(nn.Module):
    def __init__(
        self, input_size, hidden_sizes, output_size, dropout_rate, use_batchnorm
    ):
        super(DNN, self).__init__()
        layers = []
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # เลือกใช้ Tanh หรือ ReLU ตามเงื่อนไข
            if i % 2 == 0:  # ใช้ Tanh ในเลเยอร์คู่
                layers.append(nn.Tanh())
            else:  # ใช้ ReLU ในเลเยอร์คี่
                layers.append(nn.ReLU())
            
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def plot_learning_curve(train_losses, test_accuracies, features_format):
    plt.figure(figsize=(14, 6))

    # Plot Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.title(f"Loss Curve - {features_format}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy", color="green")
    plt.title(f"Accuracy Curve - {features_format}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


for feature1, feature2 in itertools.combinations(
    [
        "EDA_Phasic_EmotiBit",
        # "EDA_Tonic_EmotiBit",
        # "lf_PPG",
        # "hf_PPG",
        "lf_hf_PPG",
    ],
    2,
):
    features = ["SkinTemp_Emo", "BMI", f"{feature1}", f"{feature2}"]
    featuresFormat = f"{feature1}-{feature2}-BMI-SkinTemp_Emo"
    logging.info(f"Training model for features: {features}")

    # แบ่งข้อมูลเป็น train และ test
    X = df[features]
    y = df["PainLevel"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # แปลงข้อมูลเป็น Tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)
    
    

    # สร้าง DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # ฟังก์ชัน Objective สำหรับ Optuna
    def objective(trial):
        batch_size = trial.suggest_int(
            "batch_size",
            8,
            128,
            step=8,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # หาจำนวน Hidden Layers
        num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)

        # หา Node ในแต่ละ Hidden Layer
        hidden_sizes = [
            trial.suggest_int(f"hidden_size_{i+1}", 16, 256, log=True)
            for i in range(num_hidden_layers)
        ]

        # เลือกค่า Dropout และเปิด/ปิด BatchNorm
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05)
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])

        # หา Learning Rate
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        epochs = trial.suggest_int("num_epoch", 50, 250, step=10)

        # สร้างโมเดล
        model = DNN(
            len(features),
            hidden_sizes,
            [200, 200, 200, 200, 200],
            len(df["PainLevel"].unique()),
            dropout_rate,
            use_batchnorm,
        ).to(device)

        # Loss และ Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Early stopping
        best_accuracy = 0
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            accuracy = 100 * correct / total

            # Early stopping check
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_accuracy

    # สร้าง Study สำหรับการปรับ Hyperparameter
    study = optuna.create_study(direction="maximize", study_name=featuresFormat)
    study.optimize(objective, n_trials=100, n_jobs=4)

    logging.info(f"\nBest hyperparameters: {study.best_params}")
    logging.info(f"Best accuracy: {study.best_value:.2f}%\n")

    # เทรนโมเดลใหม่ด้วย Hyperparameter ที่ดีที่สุด
    best_params = study.best_params
    best_hidden_sizes = [
        best_params[f"hidden_size_{i+1}"]
        for i in range(best_params["num_hidden_layers"])
    ]
    best_learning_rate = best_params["learning_rate"]
    best_dropout_rate = best_params["dropout_rate"]
    best_use_batchnorm = best_params["use_batchnorm"]
    best_weight_decay = best_params["weight_decay"]
    best_batch_size = best_params["batch_size"]

    model = DNN(
        len(features),
        best_hidden_sizes,
        [200, 200, 200, 200, 200],
        len(df["PainLevel"].unique()),
        best_dropout_rate,
        best_use_batchnorm,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_learning_rate)

    num_epochs = best_params["num_epoch"]
    train_losses = []
    test_accuracies = []

    # Early stopping
    best_accuracy = 0
    patience = 5
    patience_counter = 0
    train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        logging.info(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%"
        )
        
    plot_learning_curve(train_losses, test_accuracies, featuresFormat)

    # ทดสอบโมเดลด้วยข้อมูลที่ไม่เคยเห็น
    test_data_path = rf"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\SR1600_1s.csv"
    test_df = pd.read_csv(test_data_path)

    # กำหนด features ที่ต้องการใช้ในการทดสอบ
    features_evaluate = features
    X_evaluate = test_df[features_evaluate]
    y_evaluate = test_df[
        "PainLevel"
    ]  # สมมุติว่าในข้อมูลทดสอบมีคอลัมน์ "PainLevel" เป็นป้ายระดับความเจ็บปวด

    # แปลงข้อมูลทดสอบเป็น Tensor
    X_evaluate_tensor = torch.tensor(X_evaluate.values, dtype=torch.float32).to(device)
    y_evaluate_tensor = torch.tensor(y_evaluate.values, dtype=torch.long).to(device)
    evaluate_dataset = TensorDataset(X_evaluate_tensor, y_evaluate_tensor)
    evaluate_loader = DataLoader(
        evaluate_dataset, batch_size=best_batch_size, shuffle=False
    )

    # ทำนายผล
    model.eval()
    all_predictions = []
    all_labels = []

    # ปิดการคำนวณ gradient เพื่อประหยัดหน่วยความจำ
    with torch.no_grad():
        for inputs, labels in evaluate_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # คำนวณความแม่นยำ
    unseen_accuracy = accuracy_score(all_labels, all_predictions) * 100
    print(f"Unseen Accuracy: {unseen_accuracy:.2f}%")

    # # แสดงผลลัพธ์การทำนาย
    # prediction_df = pd.DataFrame(
    #     {"Actual": y_test, "Predicted": predicted_labels}  # ข้อมูลจริงจากชุดทดสอบ
    # )

    logging.info(f"Unseen Accuracy: {unseen_accuracy:.2f}%")

    # เซฟโมเดลหลังจากทดสอบเสร็จ
    # model_path = rf"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\MetaData\Model\BestDNNModelSMOTE_NonNormalization_{featuresFormat}.pt"
    params_path = rf"D:\OneFile\WorkOnly\AllCode\Python\DeepLearning\F4_5HL_Fic\BestDNNParamsSMOTE_NonNormalization_{featuresFormat}.csv"

    # logging.info(f"Saving model to {model_path}")
    # torch.save(model, model_path)

    logging.info(f"Saving best parameters to {params_path}")
    best_params_df = pd.DataFrame(
        [
            {
                "features": featuresFormat,
                **best_params,
                # 'hidden_sizes': [200, 200, 200, 200, 200],
                "best_accuracy": best_accuracy,
                "unseen_accuracy": unseen_accuracy,
            }
        ]
    )
    best_params_df.to_csv(params_path, index=False)
