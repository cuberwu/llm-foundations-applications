import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
torch.manual_seed(SEED)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

X, y = load_digits(return_X_y=True)
X = X.astype(np.float32) / 16.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, hidden=(32,), activation="sigmoid", dropout=0.0, batch_norm=False):
        super().__init__()
        activation_layer = {
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }[activation]
        layers = []
        previous_size = 64
        for hidden_size in hidden:
            layers.append(nn.Linear(previous_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_layer())
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


def train(model, optimizer_name="sgd", learning_rate=0.1, epochs=30, weight_decay=0.0):
    optimizer_class = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[optimizer_name]
    optimizer = optimizer_class(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": []}
    start = time.perf_counter()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(X_test).argmax(dim=1)
            accuracy = (predictions == y_test).float().mean().item()
        history["loss"].append(loss.item())
        history["accuracy"].append(accuracy)

    return history, time.perf_counter() - start


def save_curves(history, filename, experiment_name):
    epochs = range(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["loss"])
    axes[0].set(title=f"{experiment_name}损失曲线", xlabel="Epoch", ylabel="交叉熵损失")
    axes[1].plot(epochs, np.array(history["accuracy"]) * 100)
    axes[1].set(title=f"{experiment_name}准确率曲线", xlabel="Epoch", ylabel="测试准确率（%）")
    figure.tight_layout()
    figure.savefig(Path(__file__).parent / "results" / filename, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    assert X_train.shape == (1437, 64) and X_test.shape == (360, 64)
    assert y_train.dtype == torch.long and y_test.dtype == torch.long

    histories = []
    elapsed_times = []
    for seed in (42, 43, 44):
        torch.manual_seed(seed)
        model = MLP(hidden=(128, 128), activation="relu", batch_norm=True)
        history, elapsed = train(
            model, optimizer_name="adam", learning_rate=0.001, epochs=30
        )
        histories.append(history)
        elapsed_times.append(elapsed)
        print(
            f"种子 {seed} - 测试准确率: {history['accuracy'][-1] * 100:.2f}%，"
            f"训练耗时: {elapsed:.3f}s"
        )

    assert all(len(history["loss"]) == 30 for history in histories)
    mean_history = {
        metric: np.mean([history[metric] for history in histories], axis=0)
        for metric in ("loss", "accuracy")
    }
    final_accuracies = np.array([history["accuracy"][-1] for history in histories])
    save_curves(mean_history, "result_exp8.png", "实验8：最优组合（三次平均）")
    print(f"平均测试准确率: {final_accuracies.mean() * 100:.2f}%")
    print(f"准确率标准差: {final_accuracies.std(ddof=1) * 100:.2f}个百分点")
    print(f"平均训练耗时: {np.mean(elapsed_times):.3f}s")
