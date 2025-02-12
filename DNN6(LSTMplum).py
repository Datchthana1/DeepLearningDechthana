import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
# from LSTM_preprocessing import DataPreprocessor
import numpy as np

# สร้างโมเดล LSTM for classification
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers, # จำนวนชั้นของ LSTM
            batch_first=True,
            dropout=dropout, # เอาไว้ป้องกัน overfitting
        )
        self.dropout = nn.Dropout(p=dropout) # Dropout after LSTM
        self.batch_norm = nn.BatchNorm1d(hidden_size) # ลดการกระจายของค่าระหว่างฝึก
        self.output_layer = nn.Linear(hidden_size, output_size) # แปลงข้อมูลจาก hidden state ของ LSTM ให้เป็นคลาสที่ต้องการทำนาย

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :] # ใช้เฉพาะ output ตัวสุดท้ายของ sequence
        # output = self.batch_norm(output)
        output = self.dropout(output)  # Apply dropout before final layer
        predictions = self.output_layer(output)
        return predictions


# func for คำนวณ matric สำหรับประเมินโมเดล
def calculate_metrics(y_true, y_pred):
    return {
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=1),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=1),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=1)
        # weighted average คำนวณค่าเฉลี่ยโดยให้น้ำหนักแต่ละคลาสตามจำนวนตัวอย่างในคลาสนั้นๆ
        # zero_division=1: กำหนดค่าที่จะใช้เมื่อเกิดการหารด้วยศูนย์
    }


def print_detailed_metrics(y_true, y_pred):
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, # แยกผลตาม class
                                target_names=["No_Requests",
                                              "Person_1_Requests",
                                              "Person_2_Requests",
                                              "Person_1_and_Person_2_Requests"],
                                zero_division=1))


# func for train model each fold (ของ cross-validation)
def train_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    """
    model: โมเดล LSTM ที่สร้างไว้
    train_loader: ข้อมูลสำหรับเทรน แบ่งเป็น batch
    val_loader: ข้อมูลสำหรับ validation
    criterion: ฟังก์ชันคำนวณ loss (ในที่นี้ใช้ CrossEntropyLoss)
    optimizer: ตัวปรับค่า weight (ในที่นี้ใช้ Adam)
    device: GPU หรือ CPU (ในที่นี้ใช้ GPU)
    epochs: จำนวนรอบการเทรน
    """
    for epoch in range(epochs):
        """
        each epoch จะมี:
            - การเทรนโมเดล(training phase)
            - การทดสอบบน validation set
            - การคำนวณและแสดงผลค่า loss และเมตริกต่างๆ
        """
        model.train() # เปลี่ยนโมเดลเป็นโหมด train
        train_loss = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward() # ปรับค่า weight ด้วย backpropagation
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader) # เก็บค่า loss

        # Validation phase
        model.eval() # เปลี่ยนโมเดลเป็นโหมด eval
        all_predictions = []
        all_labels = []
        val_loss = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        val_loss = val_loss / len(val_loader) # เก็บค่า loss
        val_metrics = calculate_metrics(all_labels, all_predictions)

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}, "
              f"Val Precision: {val_metrics['precision']:.4f}, "
              f"Val Recall: {val_metrics['recall']:.4f}")

    return val_metrics


# ประเมินโมเดลบน test set
def evaluate_model(model, test_loader, device):
    model.eval() # เปลี่ยนโมเดลเป็นโหมด eval
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    print("\nTest Set Results:")
    print_detailed_metrics(all_labels, all_predictions)
    return calculate_metrics(all_labels, all_predictions)


def main():
    print("Starting training process...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # กำหนด manual ตามค่า Default parameters ของ LSTM
    batch_size = 32
    hidden_size = 50
    num_layers = 5
    learning_rate = 0.001
    epochs = 50
    dropout = 0
    n_splits = 5 # (จำนวน fold ใน cross-validation)

    # Initialize preprocessor and get train/test split
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor(sequence_length=60, n_splits= n_splits, test_size=0.2)
    X_train_full, X_test, y_train_full, y_test = preprocessor.get_train_test_data()
    preprocessor.print_split_summary(X_train_full, X_test, y_train_full, y_test)

    # สร้าง k-fold จากข้อมูล train
    fold_data = preprocessor.get_k_fold_data(X_train_full, y_train_full)

    print(f"\nTraining with {n_splits}-fold cross validation")
    print("-" * 50)

    # Store metrics for each fold
    fold_metrics = []
    best_model = None
    best_val_f1 = 0

    # วน loop ผ่านแต่ละ fold -> each fold จะได้ F1-score จากการทดสอบบน validation set
    for fold, (X_train, X_val, y_train, y_val) in enumerate(fold_data, 1):
        print(f"\nTraining Fold {fold}/{n_splits}")
        print("-" * 30)

        # Prepare data loaders
        train_data = torch.tensor(X_train[:, :, :-1], dtype=torch.float32)
        train_labels = torch.tensor(y_train, dtype=torch.long)
        val_data = torch.tensor(X_val[:, :, :-1], dtype=torch.float32)
        val_labels = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(TensorDataset(train_data, train_labels),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data, val_labels),
                                batch_size=batch_size, shuffle=False)

        # สร้าง model (LSTM) เมื่ออยู่ใน loop k-fold -> สร้างโมเดลใหม่ทุกครั้งที่เริ่ม fold ใหม่
        model = LSTMModel(
            input_size=125,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=4, # จำนวนผลลัพธ์ (class) ที่จะทำนาย
            dropout=dropout
        ).to(device)

        criterion = nn.CrossEntropyLoss() # คำนวณค่า loss function (ผิด/มั่นใจต่ำ = ค่าจะสูง)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # เทรนโมเดลด้วย train_fold
        metrics = train_fold(model, train_loader, val_loader, criterion, optimizer, device, epochs)
        fold_metrics.append(metrics)

        # เก็บค่าเมตริกและโมเดลที่ดีที่สุดระหว่าง train
        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            best_model = model.state_dict() # เก็บ weights ของ fold ที่ได้ F1-score สูงสุด(โมเดลที่ดีที่สุด)

        # Print fold results
        print(f"\nFold {fold} Results:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"dropout: {dropout}")

    # คำนวณค่าเฉลี่ยของ metrics จากทุก fold
    avg_metrics = {
        'f1': np.mean([m['f1'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics])
    }

    print("\n" + "=" * 50)
    print("FINAL CROSS-VALIDATION RESULTS")
    print("=" * 50)
    print(f"Average F1 Score: {avg_metrics['f1']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f}")
    print("=" * 50)

    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    test_data = torch.tensor(X_test[:, :, :-1], dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(test_data, test_labels),
                             batch_size=batch_size, shuffle=False)

    # สร้างโมเดลใหม่อีกครั้ง เพื่อทดสอบกับ test set
    final_model = LSTMModel(
        input_size=125,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=4,
        dropout=dropout
    ).to(device)
    final_model.load_state_dict(best_model) # โหลด weights ที่ดีที่สุดจากการ train (best fold)
    test_metrics = evaluate_model(final_model, test_loader, device) # ทดสอบกับ test set

    print("\nFINAL TEST SET RESULTS")
    print("=" * 50)
    print(f"Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")


if __name__ == "__main__":
    main()