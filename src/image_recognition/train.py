import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .dataset import make_loaders
from .model import build_model


def evaluate(model, dl, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, val_dl, classes = make_loaders(args.data_root, args.img_size, args.batch_size, args.num_workers)
    model = build_model(num_classes=len(classes), backbone=args.backbone, pretrained=True).to(device)

    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        sched.step()

        v_loss, v_acc = evaluate(model, val_dl, device)
        print(f"[val] loss={v_loss:.4f} acc={v_acc:.4f}")

        # save best
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "backbone": args.backbone,
                "img_size": args.img_size,
            }, out_dir / "pillpal_best.pth")
            print(f"[✓] Saved new best model (acc={best_acc:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="data/processed/<dataset>")
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "efficientnet_b0"])
    args = ap.parse_args()
    train(args)