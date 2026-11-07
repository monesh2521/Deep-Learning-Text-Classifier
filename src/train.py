from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sentiment_dataset.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
assert df["text"].notna().all(), "Dataset contains empty text."
assert df.groupby("sentiment")["label"].nunique().max() == 1, "Inconsistent labels."
assert set(df["label"].unique()) == {0, 1}, "Expected labels 0 and 1."

print("Dataset size:", len(df))
print("\nClass distribution:")
print(df.groupby(["sentiment", "label"]).size())

# One fixed stratified split shared by both models.
X_train, X_test, y_train, y_test = train_test_split(
    df["text"].astype(str),
    df["label"].astype(int),
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"]
)

# Baseline: Logistic Regression
baseline = make_pipeline(
    TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True),
    LogisticRegression(max_iter=2000, C=2.0)
)
baseline.fit(X_train, y_train)
base_pred = baseline.predict(X_test)
base_acc = accuracy_score(y_test, base_pred)

print("\n========== LOGISTIC REGRESSION BASELINE ==========")
print(f"Accuracy: {base_acc:.3f}")
print(classification_report(y_test, base_pred, labels=[0, 1],
      target_names=["Negative", "Positive"], zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, base_pred, labels=[0, 1]))

# Neural network: vectorizer is fit only on training text to avoid data leakage.
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
xtr_np = vectorizer.fit_transform(X_train).toarray().astype("float32")
xte_np = vectorizer.transform(X_test).toarray().astype("float32")

xtr = torch.tensor(xtr_np)
xte = torch.tensor(xte_np)
ytr = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)

class SentimentNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = SentimentNet(xtr.shape[1])
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# Hold out validation data from training data for early stopping.
idx = np.arange(len(y_train))
tr_idx, val_idx = train_test_split(
    idx, test_size=0.20, random_state=SEED, stratify=y_train
)
x_sub = xtr[tr_idx]
y_sub = ytr[tr_idx]
x_val = xtr[val_idx]
y_val = ytr[val_idx]

best_val = float("inf")
best_state = None
patience = 15
wait = 0

for epoch in range(1, 151):
    model.train()
    optimizer.zero_grad()
    train_loss = loss_fn(model(x_sub), y_sub)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(x_val), y_val).item()

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | train loss {train_loss.item():.4f} | val loss {val_loss:.4f}")
    if wait >= patience:
        print("Early stopping at epoch", epoch)
        break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    nn_prob = torch.sigmoid(model(xte)).numpy().ravel()
nn_pred = (nn_prob >= 0.5).astype(int)
nn_acc = accuracy_score(y_test, nn_pred)

print("\n========== PYTORCH NEURAL NETWORK ==========")
print(f"Accuracy: {nn_acc:.3f}")
print(classification_report(y_test, nn_pred, labels=[0, 1],
      target_names=["Negative", "Positive"], zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, nn_pred, labels=[0, 1]))

torch.save(model.state_dict(), OUT / "sentiment_model.pth")
joblib.dump(vectorizer, OUT / "tfidf_vectorizer.joblib")
joblib.dump(baseline, OUT / "logistic_regression_baseline.joblib")

with (OUT / "metrics.txt").open("w", encoding="utf-8") as f:
    f.write(f"Dataset size: {len(df)}\n")
    f.write(f"Logistic Regression accuracy: {base_acc:.3f}\n")
    f.write(f"PyTorch accuracy: {nn_acc:.3f}\n\n")
    f.write("Logistic Regression report:\n")
    f.write(classification_report(y_test, base_pred, labels=[0,1],
              target_names=["Negative","Positive"], zero_division=0))
    f.write("\nPyTorch report:\n")
    f.write(classification_report(y_test, nn_pred, labels=[0,1],
              target_names=["Negative","Positive"], zero_division=0))

print("\nSaved model files and metrics in:", OUT)
print("Note: this is an educational sample dataset; report the measured scores, not an assumed accuracy.")
