"""Train the ANN on the Breast Cancer dataset and print the results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.ann import NeuralNetwork
from src.data import load_data, standardize

X, y, features = load_data()
print(f"Dataset: {len(X)} rows, {len(features)} features, classes: 0 = malignant, 1 = benign")

# 60% train / 20% validation / 20% test, keeping the class balance
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
# Scale using the training data only (the same scaling is applied to val and test)
X_train_s, X_val_s = standardize(X_train, X_val)
_, X_test_s = standardize(X_train, X_test)

model = NeuralNetwork(hidden_layers=(16, 8), learning_rate=0.05, seed=0)
model.fit(X_train_s, y_train, epochs=100, batch_size=32, X_val=X_val_s, y_val=y_val)

print(f"\nFinal training loss:   {model.loss_history[-1]:.4f}")
print(f"Final validation loss: {model.val_loss_history[-1]:.4f}")
print(f"Training accuracy:     {model.train_acc_history[-1]:.4f}")
print(f"Validation accuracy:   {model.val_acc_history[-1]:.4f}")

predictions = model.predict(X_test_s)
print(f"\nTest accuracy (our ANN):            {model.score(X_test_s, y_test):.4f}")
reference = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=0)
reference.fit(X_train_s, y_train)
print(f"Test accuracy (scikit-learn MLP):   {reference.score(X_test_s, y_test):.4f}")

labels = [0, 1]
cm = confusion_matrix(y_test, predictions, labels=labels)
print("\nConfusion matrix (rows = true class, columns = predicted; 0 = malignant, 1 = benign):")
print(cm)
print("\n" + classification_report(y_test, predictions, target_names=["malignant", "benign"]))

# Save the training curves
Path("results").mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(model.loss_history, label="train")
axes[0].plot(model.val_loss_history, label="validation")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(model.train_acc_history, label="train")
axes[1].plot(model.val_acc_history, label="validation")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy"); axes[1].legend()
fig.tight_layout()
fig.savefig("results/training_curves.png", dpi=150)
print("Saved plot to results/training_curves.png")
