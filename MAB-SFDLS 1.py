
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_erosion
from scipy.spatial import cKDTree

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam
except Exception:
    torch = None
    nn = None
    F = None
    Dataset = object
    DataLoader = None
    Adam = None


# -------------------------
# Paper Configuration
# -------------------------
config = {
    "dataset": "MRBrainS18",
    "total_subjects": 30,
    "train_subjects": 24,
    "test_subjects": 6,
    "modalities": ["T1", "T1_IR", "T2_FLAIR"],
    "classes": {"background": 0, "GM": 1, "WM": 2, "CSF": 3},
    "batch_size": 4,
    "patch_size": (256, 224),
    "median_image_size": (240, 204),
    "epochs": 1000,
    "contour_iterations": 2500,
    "lr": 1e-3,
    "device": "cuda" if torch is not None and torch.cuda.is_available() else "cpu",
    "save_interval": 50,
    "model_path": "Models/MAB_SFDLS_MRBrainS18",
    # Tunable because paper does not provide exact numeric values
    "mab_k": 0.5,
    "mab_sigma": 3.0,
    "mab_mu": 0.2,
    "mab_nu": 0.003,
    "mab_dt": 0.1,
    "mab_tol": 1e-5,
}


# -------------------------
# Utility Functions
# -------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_nifti(path: str) -> np.ndarray:
    if nib is None:
        raise ImportError("Please install nibabel: pip install nibabel")
    return np.asarray(nib.load(path).get_fdata())


def zscore_normalize(volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    volume = volume.astype(np.float32)
    valid = volume[mask > 0] if mask is not None and mask.any() else volume[volume > 0]
    if valid.size == 0:
        valid = volume.reshape(-1)
    return (volume - valid.mean()) / (valid.std() + 1e-8)


def resize_slice(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img.astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def merge_mrbrains_labels(label: np.ndarray) -> np.ndarray:
    """Paper mapping: GM=(1+2), WM=(3+4), CSF=(5+6), exclude 7 and 8."""
    out = np.zeros_like(label, dtype=np.uint8)
    out[np.isin(label, [1, 2])] = 1
    out[np.isin(label, [3, 4])] = 2
    out[np.isin(label, [5, 6])] = 3
    out[np.isin(label, [7, 8])] = 0
    return out


# -------------------------
# Dataset Loader
# -------------------------
class MRBrainS18SliceDataset(Dataset):
    """
    Expected folder layout:
    data_root/
      subject_01/T1.nii.gz, T1_IR.nii.gz, T2_FLAIR.nii.gz, labels.nii.gz, brain_mask.nii.gz(optional)
      subject_02/...
    """
    def __init__(self, data_root: str, split: str = "train", patch_size: Tuple[int, int] = (256, 224)):
        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        subjects = sorted([p for p in self.data_root.iterdir() if p.is_dir()])
        subjects = subjects[:config["total_subjects"]]
        if split == "train":
            subjects = subjects[:config["train_subjects"]]
        else:
            subjects = subjects[config["train_subjects"]:config["train_subjects"] + config["test_subjects"]]

        for subj in tqdm(subjects, desc=f"Loading {split} subjects"):
            t1 = load_nifti(str(subj / "T1.nii.gz"))
            t1ir = load_nifti(str(subj / "T1_IR.nii.gz"))
            flair = load_nifti(str(subj / "T2_FLAIR.nii.gz"))
            label = merge_mrbrains_labels(load_nifti(str(subj / "labels.nii.gz")).astype(np.uint8))
            mask_path = subj / "brain_mask.nii.gz"
            mask = load_nifti(str(mask_path)) > 0 if mask_path.exists() else (t1 > 0)

            t1 = zscore_normalize(t1, mask)
            t1ir = zscore_normalize(t1ir, mask)
            flair = zscore_normalize(flair, mask)

            for z in range(t1.shape[2]):
                if np.sum(mask[:, :, z]) < 10:
                    continue
                image = np.stack([
                    resize_slice(t1[:, :, z], patch_size),
                    resize_slice(t1ir[:, :, z], patch_size),
                    resize_slice(flair[:, :, z], patch_size),
                ], axis=0)
                lab = cv2.resize(label[:, :, z].astype(np.uint8), (patch_size[1], patch_size[0]), interpolation=cv2.INTER_NEAREST)
                self.samples.append((image.astype(np.float32), lab.astype(np.int64)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# -------------------------
# Deep Baseline Models
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """U-Net baseline mentioned in the paper."""
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        self.e1 = ConvBlock(in_channels, 32)
        self.e2 = ConvBlock(32, 64)
        self.e3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, num_classes, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class SimpleTransUNet(nn.Module):
    """Lightweight TransUNet-style baseline: CNN encoder + transformer encoder + decoder."""
    def __init__(self, in_channels=3, num_classes=4, embed_dim=128, heads=4, layers=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, 3, stride=4, padding=1), nn.ReLU(inplace=True)
        )
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, 4, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
    def forward(self, x):
        feat = self.stem(x)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        feat = tokens.transpose(1, 2).reshape(b, c, h, w)
        out = self.decoder(feat)
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


class UMambaLike(nn.Module):
    """U-Mamba-style placeholder using depthwise state-space-like gated convolution blocks."""
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.Conv2d(32, 32, 7, padding=3, groups=32), nn.SiLU(),
            ConvBlock(32, 64),
            nn.Conv2d(64, 64, 7, padding=3, groups=64), nn.SiLU(),
            nn.Conv2d(64, num_classes, 1)
        )
    def forward(self, x):
        return self.net(x)


# -------------------------
# Metrics
# -------------------------
def dice_score(pred, target, cls):
    pred_c, target_c = pred == cls, target == cls
    inter = np.logical_and(pred_c, target_c).sum()
    return (2 * inter + 1e-5) / (pred_c.sum() + target_c.sum() + 1e-5)


def iou_score(pred, target, cls):
    pred_c, target_c = pred == cls, target == cls
    inter = np.logical_and(pred_c, target_c).sum()
    union = np.logical_or(pred_c, target_c).sum()
    return (inter + 1e-5) / (union + 1e-5)


def sensitivity_specificity(pred, target, cls):
    pred_c, target_c = pred == cls, target == cls
    tp = np.logical_and(pred_c, target_c).sum()
    tn = np.logical_and(~pred_c, ~target_c).sum()
    fp = np.logical_and(pred_c, ~target_c).sum()
    fn = np.logical_and(~pred_c, target_c).sum()
    return (tp + 1e-5) / (tp + fn + 1e-5), (tn + 1e-5) / (tn + fp + 1e-5)


def hd95(pred_mask, gt_mask, spacing=(1.0, 1.0)):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    if not pred_mask.any() or not gt_mask.any():
        return np.nan
    pred_border = pred_mask ^ binary_erosion(pred_mask)
    gt_border = gt_mask ^ binary_erosion(gt_mask)
    pred_pts = np.argwhere(pred_border) * np.array(spacing)
    gt_pts = np.argwhere(gt_border) * np.array(spacing)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return np.nan
    d1 = cKDTree(gt_pts).query(pred_pts, k=1)[0]
    d2 = cKDTree(pred_pts).query(gt_pts, k=1)[0]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def evaluate_segmentation(pred: np.ndarray, target: np.ndarray) -> Dict[str, Dict[str, float]]:
    names = {1: "GM", 2: "WM", 3: "CSF"}
    results = {}
    for cls, name in names.items():
        sen, spe = sensitivity_specificity(pred, target, cls)
        results[name] = {
            "Dice": float(dice_score(pred, target, cls)),
            "IoU": float(iou_score(pred, target, cls)),
            "Accuracy": float((pred == target).mean()),
            "Sensitivity": float(sen),
            "Specificity": float(spe),
            "HD95": hd95(pred == cls, target == cls),
        }
    return results


# -------------------------
# Proposed MAB-SFDLS Model
# -------------------------
def heaviside(phi, eps=1.0):
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / eps))


def dirac(phi, eps=1.0):
    return (eps / np.pi) / (eps * eps + phi * phi)


def curvature(phi):
    gy, gx = np.gradient(phi)
    norm = np.sqrt(gx * gx + gy * gy) + 1e-8
    nx, ny = gx / norm, gy / norm
    nxy, _ = np.gradient(nx)
    _, nyx = np.gradient(ny)
    return nxy + nyx


class MABSFDLS:
    """Multiplicative-Additive Bias Single-Function Dual-Level-Set model."""
    def __init__(self, k=0.5, sigma=3.0, mu=0.2, nu=0.003, dt=0.1, tol=1e-5, max_iter=1000):
        self.k = k
        self.sigma = sigma
        self.mu = mu
        self.nu = nu
        self.dt = dt
        self.tol = tol
        self.max_iter = max_iter
        self.history = {"loss": [], "pseudo_dice": []}

    def memberships(self, phi):
        h0 = heaviside(phi)
        hk = heaviside(phi - self.k)
        m1 = hk
        m2 = h0 - hk
        m3 = 1.0 - h0
        return [m1, m2, m3]

    def initialize_phi(self, image):
        img = (image - image.min()) / (image.max() - image.min() + 1e-8)
        q1, q2 = np.quantile(img[img > 0], [0.33, 0.66]) if (img > 0).any() else (0.33, 0.66)
        phi = np.zeros_like(img, dtype=np.float32)
        phi[img > q2] = self.k + 1.0
        phi[(img > q1) & (img <= q2)] = self.k / 2.0
        phi[img <= q1] = -1.0
        return gaussian_filter(phi, 1.0)

    def fit_predict(self, image2d: np.ndarray, gt: Optional[np.ndarray] = None) -> np.ndarray:
        I = image2d.astype(np.float32)
        I = (I - I.min()) / (I.max() - I.min() + 1e-8)
        phi = self.initialize_phi(I)
        bm = np.ones_like(I, dtype=np.float32)
        ba = 0.0
        c = np.array([0.75, 0.45, 0.15], dtype=np.float32)

        prev_phi = phi.copy()
        for it in range(self.max_iter):
            M = self.memberships(phi)

            # update region constants ci
            for i in range(3):
                w = M[i]
                numerator = np.sum(w * bm * (I - ba))
                denominator = np.sum(w * bm * bm) + 1e-8
                c[i] = numerator / denominator

            # update multiplicative bias field bm, smoothed by Gaussian kernel K
            num = np.zeros_like(I)
            den = np.zeros_like(I)
            for i in range(3):
                num += M[i] * c[i] * (I - ba)
                den += M[i] * c[i] * c[i]
            bm = gaussian_filter(num, self.sigma) / (gaussian_filter(den, self.sigma) + 1e-8)
            bm = np.clip(bm, 0.2, 5.0)

            # update global additive bias ba
            num_ba, den_ba = 0.0, 0.0
            for i in range(3):
                num_ba += np.sum(M[i] * (I - bm * c[i]))
                den_ba += np.sum(M[i])
            ba = float(num_ba / (den_ba + 1e-8))

            e = [(I - bm * c[i] - ba) ** 2 for i in range(3)]
            d0 = dirac(phi)
            dk = dirac(phi - self.k)
            data_force = -((e[1] - e[2]) * d0 + (e[0] - e[1]) * dk)
            smooth_force = self.nu * curvature(phi)
            gy, gx = np.gradient(phi)
            dist_reg = self.mu * (np.sqrt(gx * gx + gy * gy + 1e-8) - 1.0)
            phi = phi + self.dt * (data_force + smooth_force - dist_reg)

            loss = sum(float(np.sum(M[i] * e[i])) for i in range(3))
            self.history["loss"].append(loss)
            if gt is not None:
                pred_tmp = self.label_from_phi(phi)
                self.history["pseudo_dice"].append(float(np.mean([dice_score(pred_tmp, gt, c_) for c_ in [1, 2, 3]])))

            rel_change = np.mean(np.abs(phi - prev_phi)) / (np.mean(np.abs(prev_phi)) + 1e-8)
            if rel_change < self.tol:
                break
            prev_phi = phi.copy()

        return self.label_from_phi(phi)

    def label_from_phi(self, phi):
        seg = np.zeros_like(phi, dtype=np.uint8)
        seg[phi > self.k] = 2       # WM-like high intensity region
        seg[(phi > 0) & (phi <= self.k)] = 1  # GM-like middle region
        seg[phi <= 0] = 3           # CSF/background-like low intensity region
        return seg


# -------------------------
# Training and Evaluation
# -------------------------
def train_deep_model(model, train_loader, val_loader, epochs=1000, lr=1e-3, device="cpu"):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "pseudo_dice": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        dices = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item()
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                true = masks.cpu().numpy()
                for p, t in zip(pred, true):
                    dices.append(np.mean([dice_score(p, t, cls) for cls in [1, 2, 3]]))
        val_loss /= max(len(val_loader), 1)
        pseudo_dice = float(np.mean(dices)) if dices else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["pseudo_dice"].append(pseudo_dice)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Pseudo-Dice={pseudo_dice:.4f}")
    return history


def run_mab_sfdls(data_root: str, epochs: int):
    test_set = MRBrainS18SliceDataset(data_root, split="test", patch_size=config["patch_size"])
    model = MABSFDLS(k=config["mab_k"], sigma=config["mab_sigma"], mu=config["mab_mu"],
                     nu=config["mab_nu"], dt=config["mab_dt"], tol=config["mab_tol"], max_iter=epochs)
    all_results = []
    for idx in tqdm(range(len(test_set)), desc="MAB-SFDLS testing"):
        x, y = test_set[idx]
        # Use multimodal fusion by averaging normalized T1, T1-IR and T2-FLAIR slices.
        fused = x.numpy().mean(axis=0)
        pred = model.fit_predict(fused, y.numpy())
        all_results.append(evaluate_segmentation(pred, y.numpy()))
    ensure_dir(config["model_path"])
    with open(os.path.join(config["model_path"], "mab_sfdls_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved:", os.path.join(config["model_path"], "mab_sfdls_results.json"))


def run_deep_baseline(data_root: str, model_name: str, epochs: int):
    train_set = MRBrainS18SliceDataset(data_root, split="train", patch_size=config["patch_size"])
    test_set = MRBrainS18SliceDataset(data_root, split="test", patch_size=config["patch_size"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    if model_name == "unet":
        model = UNet2D(in_channels=3, num_classes=4)
    elif model_name == "transunet":
        model = SimpleTransUNet(in_channels=3, num_classes=4)
    elif model_name == "umamba":
        model = UMambaLike(in_channels=3, num_classes=4)
    else:
        raise ValueError("model_name must be one of: unet, transunet, umamba, mab")

    history = train_deep_model(model, train_loader, test_loader, epochs=epochs, lr=config["lr"], device=config["device"])
    ensure_dir(config["model_path"])
    torch.save(model.state_dict(), os.path.join(config["model_path"], f"{model_name}_mrbrains18.pth"))
    with open(os.path.join(config["model_path"], f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MRBrainS18 MAB-SFDLS and baseline segmentation code")
    parser.add_argument("--data_root", type=str, required=True, help="Path to MRBrainS18 folder")
    parser.add_argument("--model", type=str, default="mab", choices=["mab", "unet", "transunet", "umamba"])
    parser.add_argument("--epochs", type=int, default=config["epochs"], help="Paper convergence uses 1000 epochs/iterations")
    args = parser.parse_args()

    ensure_dir(config["model_path"])
    if args.model == "mab":
        run_mab_sfdls(args.data_root, args.epochs)
    else:
        run_deep_baseline(args.data_root, args.model, args.epochs)


if __name__ == "__main__":
    main()
