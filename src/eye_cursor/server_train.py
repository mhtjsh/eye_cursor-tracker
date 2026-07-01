from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .features import feature_dim


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "Server training requires PyTorch. On the server, install with: "
            "python -m pip install -e '.[server]'"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _pixel_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y_pred - y_true, axis=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ONNX gaze model on a GPU/SLURM server.")
    parser.add_argument("--calibration", required=True, help="Path to calibration .npz.")
    parser.add_argument("--output", default="model.onnx", help="Output ONNX model path.")
    parser.add_argument("--metadata", default="model_metadata.json", help="Output metadata JSON path.")
    parser.add_argument("--metrics", default="metrics.json", help="Output metrics JSON path.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--patience", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch, nn, DataLoader, TensorDataset = _require_torch()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    calibration_path = Path(args.calibration).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    metrics_path = Path(args.metrics).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(calibration_path, allow_pickle=False)
    x = data["features"].astype(np.float32)
    y = data["targets"].astype(np.float32)
    screen_size = [int(v) for v in data["screen_size"]]

    if x.shape[1] != feature_dim():
        raise RuntimeError(f"Feature dimension mismatch: data has {x.shape[1]}, code expects {feature_dim()}.")
    if len(x) < 100:
        raise RuntimeError(f"Need at least 100 samples, found {len(x)}.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-6] = 1.0

    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std < 1e-6] = 1.0

    class GazeMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))
            self.register_buffer("x_std", torch.tensor(x_std, dtype=torch.float32))
            self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
            self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))
            self.net = nn.Sequential(
                nn.Linear(x.shape[1], args.hidden),
                nn.LayerNorm(args.hidden),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, 2),
            )

        def forward(self, raw_x):
            normalized_x = (raw_x - self.x_mean) / self.x_std
            normalized_y = self.net(normalized_x)
            return normalized_y * self.y_std + self.y_mean

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GazeMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=20.0)

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_x = torch.tensor(x_test, dtype=torch.float32, device=device)
    test_y = torch.tensor(y_test, dtype=torch.float32, device=device)

    best_state = None
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(test_x), test_y).detach().cpu())

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch} val_loss={val_loss:.3f} best={best_loss:.3f}")

        if stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch} best_val_loss={best_loss:.3f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_pred = model(torch.tensor(x_train, dtype=torch.float32, device=device)).detach().cpu().numpy()
        test_pred = model(test_x).detach().cpu().numpy()

    train_err = _pixel_errors(y_train, train_pred)
    test_err = _pixel_errors(y_test, test_pred)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "trainer": "eye_cursor.server_train.GazeMLP",
        "calibration_file": str(calibration_path),
        "sample_count": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "screen_size": screen_size,
        "device": str(device),
        "train_median_px": float(np.median(train_err)),
        "train_p90_px": float(np.percentile(train_err, 90)),
        "test_median_px": float(np.median(test_err)),
        "test_p90_px": float(np.percentile(test_err, 90)),
    }

    dummy = torch.zeros(1, x.shape[1], dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["screen_xy"],
        dynamic_axes={"features": {0: "batch"}, "screen_xy": {0: "batch"}},
        opset_version=17,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved ONNX: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Train median/p90: {metadata['train_median_px']:.1f}px / {metadata['train_p90_px']:.1f}px")
    print(f"Test median/p90: {metadata['test_median_px']:.1f}px / {metadata['test_p90_px']:.1f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
