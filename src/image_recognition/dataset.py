from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def make_transforms(img_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms


def make_loaders(data_root: str, img_size=224, batch_size=32, num_workers=4):
    train_tfms, val_tfms = make_transforms(img_size)
    train_ds = datasets.ImageFolder(root=f"{data_root}/train", transform=train_tfms)
    val_ds = datasets.ImageFolder(root=f"{data_root}/val", transform=val_tfms)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, train_ds.classes