import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class PrunableLinear(nn.Module):
    """Linear layer with learnable element-wise gates on weights."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Start near-open gates but not saturated.
        nn.init.constant_(self.gate_scores, 2.0)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.gates()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class SelfPruningMLP(nn.Module):
    """A simple feed-forward classifier built only from PrunableLinear layers."""

    def __init__(self, hidden_dims: Sequence[int], dropout: float = 0.2):
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one value.")

        dims = [32 * 32 * 3, *hidden_dims, 10]
        blocks: List[nn.Module] = []
        for i in range(len(dims) - 2):
            blocks.append(PrunableLinear(dims[i], dims[i + 1]))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(p=dropout))
        blocks.append(PrunableLinear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.net(x)

    def prunable_layers(self) -> Iterable[PrunableLinear]:
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ExperimentResult:
    lam: float
    test_accuracy: float
    sparsity_pct: float


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def gate_l1_penalty(model: SelfPruningMLP) -> torch.Tensor:
    penalties = [layer.gates().abs().sum() for layer in model.prunable_layers()]
    return torch.stack(penalties).sum()


def evaluate(model: SelfPruningMLP, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100.0 * correct / total


def compute_sparsity(model: SelfPruningMLP, threshold: float) -> float:
    all_gates = torch.cat([layer.gates().detach().flatten().cpu() for layer in model.prunable_layers()])
    pruned = (all_gates < threshold).sum().item()
    total = all_gates.numel()
    return 100.0 * pruned / total


def collect_all_gate_values(model: SelfPruningMLP) -> np.ndarray:
    return (
        torch.cat([layer.gates().detach().flatten().cpu() for layer in model.prunable_layers()])
        .numpy()
        .astype(np.float32)
    )


def train_one_lambda(
    lam: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dims: Sequence[int],
    dropout: float,
    sparsity_threshold: float,
) -> tuple[SelfPruningMLP, ExperimentResult]:
    model = SelfPruningMLP(hidden_dims=hidden_dims, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_sparse = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)

            classification_loss = criterion(logits, labels)
            sparsity_loss = gate_l1_penalty(model)
            total_loss = classification_loss + lam * sparsity_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_cls += classification_loss.item()
            running_sparse += sparsity_loss.item()

        avg_loss = running_loss / len(train_loader)
        avg_cls = running_cls / len(train_loader)
        avg_sparse = running_sparse / len(train_loader)
        test_acc = evaluate(model, test_loader, device)
        sparsity_pct = compute_sparsity(model, sparsity_threshold)

        print(
            f"[lambda={lam:.2e}] Epoch {epoch:02d}/{epochs} | "
            f"loss={avg_loss:.4f} cls={avg_cls:.4f} sparse={avg_sparse:.2f} "
            f"| test_acc={test_acc:.2f}% sparsity={sparsity_pct:.2f}%"
        )

    final_accuracy = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model, sparsity_threshold)

    result = ExperimentResult(
        lam=lam,
        test_accuracy=final_accuracy,
        sparsity_pct=final_sparsity,
    )
    return model, result


def save_results_csv(results: Sequence[ExperimentResult], csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Lambda", "Test Accuracy", "Sparsity Level (%)"])
        for r in results:
            writer.writerow([f"{r.lam:.2e}", f"{r.test_accuracy:.2f}", f"{r.sparsity_pct:.2f}"])


def save_gate_histogram(gate_values: np.ndarray, image_path: str, lam: float) -> None:
    os.makedirs(os.path.dirname(image_path), exist_ok=True) if os.path.dirname(image_path) else None
    plt.figure(figsize=(8, 4.5))
    plt.hist(gate_values, bins=80, range=(0.0, 1.0), color="#1f77b4", alpha=0.85)
    plt.title(f"Gate Value Distribution (best lambda={lam:.2e})")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(image_path, dpi=160)
    plt.close()


def generate_markdown_report(
    report_path: str,
    results: Sequence[ExperimentResult],
    best_lambda: float,
    best_hist_path: str,
    sparsity_threshold: float,
) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True) if os.path.dirname(report_path) else None

    lines: List[str] = []
    lines.append("# Self-Pruning Neural Network Report")
    lines.append("")
    lines.append("## Why L1 on Sigmoid Gates Encourages Sparsity")
    lines.append(
        "Each gate is defined as `g = sigmoid(s)` and constrained to `[0,1]`. "
        "Adding an L1 penalty on these gate values means the optimizer pays a direct cost "
        "for every active connection. To reduce the objective, it is incentivized to push many "
        "gates toward 0. Once a gate is near zero, the corresponding effective weight "
        "`w_eff = w * g` contributes almost nothing to the output, effectively pruning that connection."
    )
    lines.append("")
    lines.append("## Experiment Summary")
    lines.append("")
    lines.append("| Lambda | Test Accuracy | Sparsity Level (%) |")
    lines.append("|---:|---:|---:|")
    for r in results:
        lines.append(f"| {r.lam:.2e} | {r.test_accuracy:.2f}% | {r.sparsity_pct:.2f}% |")
    lines.append("")
    lines.append(f"Sparsity threshold used: `{sparsity_threshold}`")
    lines.append("")
    lines.append("## Gate Distribution Plot (Best Model)")
    lines.append("")
    lines.append(f"Best model selected by highest test accuracy, with lambda = `{best_lambda:.2e}`.")
    lines.append("")
    normalized_hist_path = best_hist_path.replace("\\", "/")
    lines.append(f"![Gate Value Distribution]({normalized_hist_path})")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-pruning neural network on CIFAR-10.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for CIFAR-10 download/cache.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs per lambda run.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for train/test loaders.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout applied in hidden blocks.")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 512], help="Hidden layer sizes.")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e-6, 5e-6, 2e-5],
        help="Lambda values for sparsity regularization experiments.",
    )
    parser.add_argument(
        "--sparsity-threshold",
        type=float,
        default=1e-2,
        help="Gate values below this threshold are counted as pruned.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--results-csv", type=str, default="results/summary.csv", help="CSV output path.")
    parser.add_argument("--report-path", type=str, default="REPORT.md", help="Markdown report output path.")
    parser.add_argument(
        "--hist-path",
        type=str,
        default="results/best_gate_hist.png",
        help="Output path for best-model gate histogram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Train samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    results: List[ExperimentResult] = []
    best_model = None
    best_acc = -1.0
    best_lambda = None

    for lam in args.lambdas:
        print("\n" + "=" * 80)
        print(f"Starting experiment with lambda={lam:.2e}")
        print("=" * 80)

        model, result = train_one_lambda(
            lam=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            sparsity_threshold=args.sparsity_threshold,
        )

        results.append(result)
        if result.test_accuracy > best_acc:
            best_acc = result.test_accuracy
            best_model = model
            best_lambda = lam

    if best_model is None or best_lambda is None:
        raise RuntimeError("No experiments were run. Check --lambdas argument.")

    gate_values = collect_all_gate_values(best_model)
    save_gate_histogram(gate_values, args.hist_path, best_lambda)
    save_results_csv(results, args.results_csv)

    report_hist_path = os.path.relpath(args.hist_path, os.path.dirname(args.report_path) or ".")
    generate_markdown_report(
        report_path=args.report_path,
        results=results,
        best_lambda=best_lambda,
        best_hist_path=report_hist_path,
        sparsity_threshold=args.sparsity_threshold,
    )

    print("\nAll experiments complete.")
    print(f"Saved CSV summary to: {args.results_csv}")
    print(f"Saved gate histogram to: {args.hist_path}")
    print(f"Saved markdown report to: {args.report_path}")


if __name__ == "__main__":
    main()
