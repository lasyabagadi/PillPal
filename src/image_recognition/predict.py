from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms


def load_checkpoint(ckpt_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet18")
    from .model import build_model
    model = build_model(num_classes=len(ckpt["classes"]), backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)

    tfm = transforms.Compose([
        transforms.Resize((ckpt.get("img_size", 224), ckpt.get("img_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, ckpt["classes"], tfm, device


def predict_image(ckpt_path: str, image_path: str, topk: int = 3):
    model, classes, tfm, device = load_checkpoint(ckpt_path)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        x = tfm(im).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze()
    topk_probs, topk_idx = torch.topk(probs, k=min(topk, len(classes)))
    results = [(classes[i], float(p)) for i, p in zip(topk_idx.tolist(), topk_probs.tolist())]
    return results

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    print(predict_image(args.ckpt, args.image, args.topk))