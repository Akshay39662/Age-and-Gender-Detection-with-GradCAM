import os
import csv
import random
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# ------------------ PATHS ------------------

BASE_DIR = r"C:\Users\aksha\OneDrive\Desktop\Age and Gender Detection 2\Data"
RAW_DIR = os.path.join(BASE_DIR, "DataSet")              # UTKFace raw images
PROC_DIR = os.path.join(BASE_DIR, "processed_advanced")  # new preprocessed dir
IMG_DIR = os.path.join(PROC_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

LABELS_CSV = os.path.join(PROC_DIR, "labels.csv")

# ------------------ CONFIG ------------------

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ------------------ UTILS ------------------

def parse_utkfile(name: str):
    """
    UTKFace format: age_gender_race_date.jpg
    """
    base = os.path.splitext(name)[0]
    parts = base.split("_")
    if len(parts) < 4:
        return None
    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except:
        return None
    if not (0 <= age <= 116):
        return None
    if gender not in [0, 1]:
        return None
    return age, gender, race

# ------------------ FACE DETECTION + CLAHE PREPROCESS ------------------

def detect_face_and_crop(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect largest face, expand box a bit, crop, apply CLAHE, resize to IMG_SIZE.
    img: BGR (cv2.imread)
    """
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    # largest face
    areas = [(w * h, (x, y, w, h)) for (x, y, w, h) in faces]
    _, (x, y, w, h) = max(areas, key=lambda t: t[0])

    # expand box by 25% to capture context
    expand = 0.25
    cx = x + w / 2
    cy = y + h / 2
    size = max(w, h)
    size = size * (1.0 + expand)

    x1 = int(cx - size / 2)
    y1 = int(cy - size / 2)
    x2 = int(cx + size / 2)
    y2 = int(cy + size / 2)

    h_img, w_img = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None

    # CLAHE on Y channel
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    face_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # resize to IMG_SIZE
    face_resized = cv2.resize(face_eq, (IMG_SIZE, IMG_SIZE))
    return face_resized

def build_preprocessed_dataset():
    """
    If labels.csv doesn't exist, run face detection + cropping + CLAHE
    on all UTKFace images and save processed faces + labels.
    """
    if os.path.exists(LABELS_CSV):
        print("labels.csv already exists, skipping preprocessing.")
        return

    rows = []
    idx = 0

    files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".jpg")]
    total = len(files)
    print(f"Starting preprocessing with Haar cascade on {total} images...")

    for i, fname in enumerate(files):
        parsed = parse_utkfile(fname)
        if parsed is None:
            continue
        age, gender, race = parsed

        img_path = os.path.join(RAW_DIR, fname)
        img = cv2.imread(img_path)
        face = detect_face_and_crop(img)
        if face is None:
            continue

        # save processed face as RGB JPG
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        new_name = f"{idx:06d}.jpg"
        pil_img.save(os.path.join(IMG_DIR, new_name), quality=95)
        rows.append({"filename": new_name, "age": age, "gender": gender, "race": race})
        idx += 1

        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{total} raw images...")

    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["filename", "age", "gender", "race"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Preprocessing complete. Saved {len(rows)} faces to {IMG_DIR}")

# ------------------ DATASET & TRANSFORMS ------------------

class UTKFaceAdvancedDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, tfm):
        with open(csv_path, newline="", encoding="utf-8") as fp:
            self.rows = list(csv.DictReader(fp))
        self.img_dir = img_dir
        self.transform = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(os.path.join(self.img_dir, r["filename"])).convert("RGB")
        img = self.transform(img)
        age = torch.tensor(float(r["age"]), dtype=torch.float32)
        gender = torch.tensor(float(r["gender"]), dtype=torch.float32)
        return img, age, gender

def get_transforms():
    w = EfficientNet_B0_Weights.IMAGENET1K_V1

    train_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        AutoAugment(AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=w.transforms().mean, std=w.transforms().std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=w.transforms().mean, std=w.transforms().std),
    ])

    return train_tfm, val_tfm

def split_csv():
    with open(LABELS_CSV, newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    random.shuffle(rows)
    n = len(rows)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    def save(path, data):
        with open(path, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=["filename", "age", "gender", "race"])
            w.writeheader()
            for r in data:
                w.writerow(r)

    train_csv = os.path.join(PROC_DIR, "train.csv")
    val_csv   = os.path.join(PROC_DIR, "val.csv")
    test_csv  = os.path.join(PROC_DIR, "test.csv")

    save(train_csv, train_rows)
    save(val_csv, val_rows)
    save(test_csv, test_rows)

    return train_csv, val_csv, test_csv

# ------------------ CBAM + MODEL ------------------

class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, c // r, 1),
            nn.ReLU(),
            nn.Conv2d(c // r, c, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, maxv], 1)))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class AgeGenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        w = EfficientNet_B0_Weights.IMAGENET1K_V1
        net = efficientnet_b0(weights=w)
        self.features = net.features
        self.cbam = CBAM(1280)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.4)

        self.gender_head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.age_head = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        g = self.gender_head(x).squeeze(1)
        a = self.age_head(x).squeeze(1)
        return g, a

# ------------------ TRAINING ------------------

def train_epoch(model, loader, opt, g_loss, a_loss):
    model.train()
    total = 0
    loss_sum = 0.0
    correct = 0
    mae = 0.0

    for imgs, ages, genders in loader:
        imgs, ages, genders = imgs.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
        opt.zero_grad()
        g_logits, a_pred = model(imgs)
        loss = g_loss(g_logits, genders) + a_loss(a_pred, ages)
        loss.backward()
        opt.step()

        bs = imgs.size(0)
        total += bs
        loss_sum += loss.item() * bs
        preds = (torch.sigmoid(g_logits) > 0.5).float()
        correct += (preds == genders).sum().item()
        mae += torch.abs(a_pred - ages).sum().item()

    return loss_sum / total, correct / total, mae / total

def eval_epoch(model, loader, g_loss, a_loss):
    model.eval()
    total = 0
    loss_sum = 0.0
    correct = 0
    mae = 0.0

    with torch.no_grad():
        for imgs, ages, genders in loader:
            imgs, ages, genders = imgs.to(DEVICE), ages.to(DEVICE), genders.to(DEVICE)
            g_logits, a_pred = model(imgs)
            loss = g_loss(g_logits, genders) + a_loss(a_pred, ages)

            bs = imgs.size(0)
            total += bs
            loss_sum += loss.item() * bs
            preds = (torch.sigmoid(g_logits) > 0.5).float()
            correct += (preds == genders).sum().item()
            mae += torch.abs(a_pred - ages).sum().item()

    return loss_sum / total, correct / total, mae / total

# ------------------ MAIN ------------------

def main():
    build_preprocessed_dataset()
    train_csv, val_csv, test_csv = split_csv()

    train_tfm, val_tfm = get_transforms()

    train_ds = UTKFaceAdvancedDataset(train_csv, IMG_DIR, train_tfm)
    val_ds   = UTKFaceAdvancedDataset(val_csv, IMG_DIR, val_tfm)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AgeGenderNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g_loss = nn.BCEWithLogitsLoss()
    a_loss = nn.L1Loss()

    best = float("inf")
    ckpt = os.path.join(PROC_DIR, "best_model_advanced.pth")

    for epoch in range(EPOCHS):
        tr_l, tr_g, tr_a = train_epoch(model, train_ld, opt, g_loss, a_loss)
        vl_l, vl_g, vl_a = eval_epoch(model, val_ld, g_loss, a_loss)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f" Train: Loss={tr_l:.4f}, GenderAcc={tr_g:.3f}, AgeMAE={tr_a:.2f}")
        print(f" Val:   Loss={vl_l:.4f}, GenderAcc={vl_g:.3f}, AgeMAE={vl_a:.2f}")

        if vl_l < best:
            best = vl_l
            torch.save(model.state_dict(), ckpt)
            print(" ✔ Saved new best model")

    print("\nTraining complete!")
    print("Best advanced model saved at:", ckpt)

if __name__ == "__main__":
    main()
