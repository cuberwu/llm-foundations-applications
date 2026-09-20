import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
N_TRAIN = 2000
N_TEST = 200
RESULTS_DIR = Path(__file__).parent / "results"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def make_sequences(n_seq=N_TRAIN + N_TEST, frames=10, size=32, radius=2, seed=SEED):
    """生成形状为 (序列数, 时间, 高, 宽, 通道) 的弹跳小球序列。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    sequences = np.zeros((n_seq, frames, size, size, 1), dtype=np.float32)
    low, high = radius, size - 1 - radius

    for sequence in sequences:
        x, y = rng.uniform(low, high, 2)
        vx, vy = rng.choice((-1, 1), 2) * rng.uniform(0.8, 1.6, 2)

        for frame in sequence:
            x += vx
            y += vy
            if x < low or x > high:
                vx = -vx
                x = np.clip(x, low, high)
            if y < low or y > high:
                vy = -vy
                y = np.clip(y, low, high)
            frame[..., 0] = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2

    return sequences


def prepare_data(input_frames=4, n_train=N_TRAIN, n_test=N_TEST, seed=SEED):
    data = make_sequences(n_train + n_test, seed=seed)
    train_data, test_data = data[:n_train], data[n_train:]
    train_x = torch.from_numpy(train_data[:, :input_frames]).permute(0, 1, 4, 2, 3)
    train_y = torch.from_numpy(train_data[:, input_frames, :, :, 0])
    test_x = torch.from_numpy(test_data[:, :input_frames]).permute(0, 1, 4, 2, 3)
    test_y = torch.from_numpy(test_data[:, input_frames, :, :, 0])
    return train_x, train_y, test_x, test_y


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x, state):
        hidden, cell = state
        input_gate, forget_gate, candidate, output_gate = self.gates(
            torch.cat((x, hidden), dim=1)
        ).chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * torch.tanh(candidate)
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ConvLSTM(nn.Module):
    def __init__(
        self, in_channels=1, hidden_channels=32, kernel_size=3, num_layers=1
    ):
        super().__init__()
        channels = [in_channels] + [hidden_channels] * num_layers
        self.cells = nn.ModuleList(
            ConvLSTMCell(channels[i], channels[i + 1], kernel_size)
            for i in range(num_layers)
        )
        self.output = nn.Conv2d(hidden_channels, 1, 3, padding=1)

    def forward(self, x):
        batch, _, _, height, width = x.shape
        states = [
            (
                x.new_zeros(batch, layer.hidden_channels, height, width),
                x.new_zeros(batch, layer.hidden_channels, height, width),
            )
            for layer in self.cells
        ]

        for time_step in range(x.shape[1]):
            layer_input = x[:, time_step]
            for index, layer in enumerate(self.cells):
                states[index] = layer(layer_input, states[index])
                layer_input = states[index][0]

        return self.output(states[-1][0]).squeeze(1)


class FlattenLSTM(nn.Module):
    """把每帧 32x32 图像展平后交给普通 LSTM。"""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.recurrent = nn.LSTM(32 * 32, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 32 * 32)

    def forward(self, x):
        batch, frames, _, height, width = x.shape
        _, (hidden, _) = self.recurrent(x.reshape(batch, frames, height * width))
        return self.output(hidden[-1]).reshape(batch, height, width)


@torch.inference_mode()
def evaluate(model, inputs, targets, batch_size=64):
    model.eval()
    predictions = torch.cat(
        [model(inputs[i : i + batch_size]) for i in range(0, len(inputs), batch_size)]
    )
    error = predictions - targets
    return error.square().mean().item(), error.abs().mean().item(), predictions


def train_model(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
    loss_function=None,
    learning_rate=0.001,
    epochs=5,
    batch_size=64,
):
    loss_function = loss_function or nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "test_mse": [], "test_mae": []}
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        permutation = torch.randperm(len(train_x))
        for offset in range(0, len(train_x), batch_size):
            indexes = permutation[offset : offset + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(train_x[indexes]), train_y[indexes])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indexes)

        mse, mae, predictions = evaluate(model, test_x, test_y, batch_size)
        history["train_loss"].append(total_loss / len(train_x))
        history["test_mse"].append(mse)
        history["test_mae"].append(mae)
        print(
            f"epoch {epoch + 1}: train_loss={history['train_loss'][-1]:.6f}, "
            f"test_MSE={mse:.6f}, test_MAE={mae:.6f}"
        )

    return history, time.perf_counter() - started, predictions


def save_curves(history, filename, title, loss_label="MSE"):
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o")
    axes[0].set(
        title=f"{title}训练损失", xlabel="Epoch", ylabel=f"{loss_label} 损失"
    )
    axes[1].plot(epochs, history["test_mse"], marker="o", color="tab:orange")
    axes[1].set(title=f"{title}测试 MSE", xlabel="Epoch", ylabel="MSE")
    for axis in axes:
        axis.set_xticks(list(epochs))
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / filename, dpi=160)
    plt.close(figure)


def save_predictions(inputs, targets, predictions, filename, indexes=(0, 50, 100)):
    columns = inputs.shape[1] + 2
    figure, axes = plt.subplots(len(indexes), columns, figsize=(2 * columns, 6))
    for row, index in enumerate(indexes):
        for frame in range(inputs.shape[1]):
            axes[row, frame].imshow(inputs[index, frame, 0], cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, frame].set_title(f"输入 {frame + 1}")
        axes[row, -2].imshow(targets[index], cmap="gray", vmin=0, vmax=1)
        axes[row, -1].imshow(predictions[index].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        sample_mse = (predictions[index] - targets[index]).square().mean().item()
        if row == 0:
            axes[row, -2].set_title("真实下一帧")
        axes[row, -1].set_title(f"预测帧\nMSE={sample_mse:.5f}")
        axes[row, 0].set_ylabel(f"样本 {index}")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / filename, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    experiments = [
        (1, "加深层数", {"num_layers": 2}, 4, "MSE"),
        (2, "隐藏通道", {"hidden_channels": 64}, 4, "MSE"),
        (3, "卷积核", {"kernel_size": 5}, 4, "MSE"),
        (4, "输入帧数", {}, 8, "MSE"),
        (5, "结构对照", {"model": "FlattenLSTM"}, 4, "MSE"),
        (6, "损失函数", {}, 4, "L1"),
    ]

    for number, name, changes, input_frames, loss_name in experiments:
        key = f"exp{number}"
        if key in metrics:
            print(f"跳过已完成的实验 {number}：{name}")
            continue

        print(f"\n开始实验 {number}：{name}")
        torch.manual_seed(SEED)
        train_x, train_y, test_x, test_y = prepare_data(input_frames=input_frames)
        if changes.get("model") == "FlattenLSTM":
            model = FlattenLSTM()
        else:
            model = ConvLSTM(
                hidden_channels=changes.get("hidden_channels", 32),
                kernel_size=changes.get("kernel_size", 3),
                num_layers=changes.get("num_layers", 1),
            )
        loss_function = nn.L1Loss() if loss_name == "L1" else nn.MSELoss()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        history, elapsed, prediction = train_model(
            model,
            train_x,
            train_y,
            test_x,
            test_y,
            loss_function=loss_function,
        )
        assert np.isfinite(history["train_loss"][-1])
        assert prediction.shape == (N_TEST, 32, 32)
        save_curves(history, f"result_exp{number}.png", f"实验 {number}：{name}", loss_name)

        metrics[key] = {
            "name": name,
            "seed": SEED,
            "config": {
                "model": changes.get("model", "ConvLSTM"),
                "layers": changes.get("num_layers", 1),
                "hidden": changes.get("hidden_channels", 32),
                "kernel": changes.get("kernel_size", 3),
                "input_frames": input_frames,
                "loss": loss_name,
            },
            "parameters": parameter_count,
            "mse": history["test_mse"][-1],
            "mae": history["test_mae"][-1],
            "seconds": elapsed,
            "history": history,
        }
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"实验 {number} 完成：参数={parameter_count}, "
            f"MSE={history['test_mse'][-1]:.6f}, "
            f"MAE={history['test_mae'][-1]:.6f}, 耗时={elapsed:.2f}s"
        )

    print("\n六组控制变量实验完成")
    for key in ("baseline", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6"):
        result = metrics[key]
        print(
            f"{key}: MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, "
            f"参数={result['parameters']}, 耗时={result['seconds']:.2f}s"
        )

    print("\n开始最优组合复测：两层 ConvLSTM + 5x5 卷积核")
    train_x, train_y, test_x, test_y = prepare_data()
    best = metrics.setdefault(
        "best",
        {
            "name": "两层 + 5x5 卷积核",
            "config": {
                "model": "ConvLSTM",
                "layers": 2,
                "hidden": 32,
                "kernel": 5,
                "input_frames": 4,
                "loss": "MSE",
            },
            "parameters": 310_945,
            "runs": [],
            "prediction_samples": [164, 59, 84],
        },
    )
    completed_seeds = {run["seed"] for run in best["runs"]}
    for seed in (42, 43, 44):
        if seed in completed_seeds:
            print(f"跳过已完成的最优组合种子 {seed}")
            continue
        print(f"最优组合种子 {seed}")
        torch.manual_seed(seed)
        model = ConvLSTM(num_layers=2, kernel_size=5)
        history, elapsed, prediction = train_model(
            model, train_x, train_y, test_x, test_y
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        assert parameter_count == best["parameters"]
        best["runs"].append(
            {
                "seed": seed,
                "mse": history["test_mse"][-1],
                "mae": history["test_mae"][-1],
                "seconds": elapsed,
                "history": history,
            }
        )
        if seed == 42:
            save_predictions(
                test_x,
                test_y,
                prediction,
                "result_pred.png",
                indexes=tuple(best["prediction_samples"]),
            )
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"种子 {seed} 完成：MSE={history['test_mse'][-1]:.6f}, "
            f"MAE={history['test_mae'][-1]:.6f}, 耗时={elapsed:.2f}s"
        )

    runs = best["runs"]
    best["mean"] = {
        metric: float(np.mean([run[metric] for run in runs]))
        for metric in ("mse", "mae", "seconds")
    }
    best["std"] = {
        metric: float(np.std([run[metric] for run in runs], ddof=1))
        for metric in ("mse", "mae", "seconds")
    }
    mean_history = {
        metric: np.mean([run["history"][metric] for run in runs], axis=0).tolist()
        for metric in ("train_loss", "test_mse", "test_mae")
    }
    save_curves(mean_history, "result_best.png", "最优组合（三次平均）")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"最优组合三次平均：MSE={best['mean']['mse']:.6f}, "
        f"MAE={best['mean']['mae']:.6f}, 平均耗时={best['mean']['seconds']:.2f}s"
    )
