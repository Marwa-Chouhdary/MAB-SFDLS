import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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
except Exception:
    torch = None
    nn = None
    F = None
    Dataset = object
    DataLoader = None


# -----------------------------
# Paper constants/configuration
# -----------------------------
PAPER_CONFIG = {
    "dataset": "MRBrainS18",
    "subjects_total": 30,
    "train_subjects": 24,
    "test_subjects": 6,
    "modalities": ["T1", "T1_IR", "T2_FLAIR"],
    "voxel_resolution_mm": [0.958, 0.958, 3.0],
    "output_classes": {"background": 0, "gray_matter": 1, "white_matter": 2, "csf": 3},
    "label_merge": {"gray_matter": [1, 2], "white_matter": [3, 4], "csf": [5, 6], "exclude": [7, 8, 9, 10]},
    "deep_batch_size": 4,
    "deep_patch_size": [256, 224],
    "median_image_size_voxels": [240.0, 204.0],
    "mab_epochs_reported": 1000,
    "visualization_iterations_reported": 2500,
    "hardware": {"CPU": "Intel i7-10700K", "RAM": "DDR4 16 GB", "GPU": "NVIDIA RTX 3060 Ti 8 GB"},
}


# -----------------------------
# I/O and preprocessing
# -----------------------------
def load_nifti(path: Path) -> np.ndarray:
    if nib is None:
        raise ImportError("Please install nibabel: pip install nibabel")
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def save_nifti_like(array: np.ndarray, reference_path: Path, out_path: Path) -> None:
    if nib is None:
        raise ImportError("Please install nibabel: pip install nibabel")
    ref = nib.load(str(reference_path))
    img = nib.Nifti1Image(array.astype(np.int16), affine=ref.affine, header=ref.header)
    nib.save(img, str(out_path))


def zscore_normalize(volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if mask is None:
        mask = volume > np.percentile(volume, 1)
    vals = volume[mask > 0]
    if vals.size == 0:
        return volume.astype(np.float32)
    return ((volume - vals.mean()) / (vals.std() + 1e-8)).astype(np.float32)


def merge_mrbrains_labels(label: np.ndarray) -> np.ndarray:
    """Merge MRBrainS18 anatomical labels into GM, WM, CSF and remove excluded labels."""
    out = np.zeros_like(label, dtype=np.uint8)
    out[np.isin(label, PAPER_CONFIG["label_merge"]["gray_matter"])] = 1
    out[np.isin(label, PAPER_CONFIG["label_merge"]["white_matter"])] = 2
    out[np.isin(label, PAPER_CONFIG["label_merge"]["csf"])] = 3
    out[np.isin(label, PAPER_CONFIG["label_merge"]["exclude"])] = 0
    return out


def find_subjects(data_root: Path) -> List[Path]:
    subjects = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    if not subjects:
        raise FileNotFoundError(f"No subject folders found under: {data_root}")
    return subjects


def locate_file(subject_dir: Path, keys: List[str]) -> Path:
    files = list(subject_dir.glob("*.nii")) + list(subject_dir.glob("*.nii.gz"))
    lower = {f.name.lower(): f for f in files}
    for f in files:
        name = f.name.lower().replace("-", "_")
        if all(k.lower() in name for k in keys):
            return f
    raise FileNotFoundError(f"Could not find file with keys {keys} in {subject_dir}")


# -----------------------------
# Metrics from paper
# -----------------------------
def binary_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return {"dice": dice, "iou": iou, "accuracy": acc, "sensitivity": sens, "specificity": spec}


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(0.958, 0.958, 3.0)) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")
    pred_border = np.logical_xor(pred, binary_erosion(pred))
    gt_border = np.logical_xor(gt, binary_erosion(gt))
    p_pts = np.argwhere(pred_border) * np.asarray(spacing)
    g_pts = np.argwhere(gt_border) * np.asarray(spacing)
    if len(p_pts) == 0 or len(g_pts) == 0:
        return float("inf")
    tree_g = cKDTree(g_pts)
    tree_p = cKDTree(p_pts)
    d_pg, _ = tree_g.query(p_pts, k=1)
    d_gp, _ = tree_p.query(g_pts, k=1)
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def evaluate_multiclass(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Dict[str, float]]:
    names = {1: "GM", 2: "WM", 3: "CSF"}
    out = {}
    for cls, name in names.items():
        stats = binary_stats(pred == cls, gt == cls)
        stats["hd95"] = hd95(pred == cls, gt == cls)
        out[name] = stats
    return out


# -----------------------------
# MAB-SFDLS implementation
# -----------------------------
def heaviside(x: np.ndarray, eps: float) -> np.ndarray:
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(x / eps))


def dirac_delta(x: np.ndarray, eps: float) -> np.ndarray:
    return (eps / np.pi) / (eps * eps + x * x)


def div2(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    nxx = np.gradient(nx, axis=0)
    nyy = np.gradient(ny, axis=1)
    return nxx + nyy


def curvature(phi: np.ndarray) -> np.ndarray:
    gx, gy = np.gradient(phi)
    norm = np.sqrt(gx * gx + gy * gy) + 1e-8
    return div2(gx / norm, gy / norm)


def distance_regularization(phi: np.ndarray) -> np.ndarray:
    gx, gy = np.gradient(phi)
    s = np.sqrt(gx * gx + gy * gy) + 1e-8
    # derivative of p(s)=0.5*(s-1)^2 gives dp=(s-1), use dp(s)/s factor
    dps = (s - 1.0) / s
    return div2(dps * gx, dps * gy)


def initialize_level_set(shape: Tuple[int, int], radius_ratio: float = 0.35) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = h * radius_ratio, w * radius_ratio
    signed = 1.0 - np.sqrt(((yy - cy) / (ry + 1e-8)) ** 2 + ((xx - cx) / (rx + 1e-8)) ** 2)
    return signed.astype(np.float32)


class MABSFDLS:
    def __init__(
        self,
        epochs: int = PAPER_CONFIG["mab_epochs_reported"],
        sigma: float = 3.0,
        mu: float = 0.2,
        nu: float = 0.003,
        timestep: float = 0.1,
        k_level: float = 0.5,
        eps: float = 1.0,
        tol: float = 1e-5,
        smooth_bm_sigma: float = 4.0,
        verbose: bool = True,
    ):
        self.epochs = epochs
        self.sigma = sigma
        self.mu = mu
        self.nu = nu
        self.timestep = timestep
        self.k = k_level
        self.eps = eps
        self.tol = tol
        self.smooth_bm_sigma = smooth_bm_sigma
        self.verbose = verbose
        self.history = {"energy": [], "change": []}

    def _memberships(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h_phi = heaviside(phi, self.eps)
        h_phik = heaviside(phi - self.k, self.eps)
        m1 = h_phik
        m2 = h_phi - h_phik
        m3 = 1.0 - h_phi
        return m1, m2, m3

    def _update_region_constants(self, I: np.ndarray, bm: np.ndarray, ba: float, memberships) -> np.ndarray:
        cs = []
        for u in memberships:
            num = gaussian_filter(bm * (I - ba) * u, self.sigma)
            den = gaussian_filter((bm * bm) * u, self.sigma) + 1e-8
            local_c = num / den
            c = np.sum(local_c * u) / (np.sum(u) + 1e-8)
            cs.append(float(c))
        return np.asarray(cs, dtype=np.float32)

    def _update_biases(self, I: np.ndarray, c: np.ndarray, memberships) -> Tuple[np.ndarray, float]:
        J1 = sum(c[i] * memberships[i] for i in range(3))
        J2 = sum((c[i] ** 2) * memberships[i] for i in range(3)) + 1e-8
        # additive bias as global offset from Eq. 17 approximation
        ba_num = sum(gaussian_filter((I - J1) * memberships[i], self.sigma).sum() for i in range(3))
        ba_den = sum(gaussian_filter(memberships[i], self.sigma).sum() for i in range(3)) + 1e-8
        ba = float(ba_num / ba_den)
        bm = gaussian_filter((I - ba) * J1, self.sigma) / (gaussian_filter(J2, self.sigma) + 1e-8)
        bm = gaussian_filter(bm, self.smooth_bm_sigma)
        bm = np.clip(bm, 0.2, 5.0).astype(np.float32)
        return bm, ba

    def _fitting_errors(self, I: np.ndarray, bm: np.ndarray, ba: float, c: np.ndarray) -> List[np.ndarray]:
        return [gaussian_filter((I - bm * c[i] - ba) ** 2, self.sigma) for i in range(3)]

    def fit_slice(self, image2d: np.ndarray, mask2d: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        I = image2d.astype(np.float32)
        if mask2d is None:
            mask2d = I > np.percentile(I, 5)
        I = zscore_normalize(I, mask2d)
        phi = initialize_level_set(I.shape)
        bm = np.ones_like(I, dtype=np.float32)
        ba = 0.0
        last_phi = phi.copy()

        for epoch in range(1, self.epochs + 1):
            memberships = self._memberships(phi)
            c = self._update_region_constants(I, bm, ba, memberships)
            bm, ba = self._update_biases(I, c, memberships)
            e1, e2, e3 = self._fitting_errors(I, bm, ba, c)

            d0 = dirac_delta(phi, self.eps)
            dk = dirac_delta(phi - self.k, self.eps)
            data_force = -dk * e1 - (d0 - dk) * e2 + d0 * e3
            length_force = self.nu * (d0 * curvature(phi) + dk * curvature(phi - self.k))
            dist_force = self.mu * distance_regularization(phi)
            dphi = data_force + length_force + dist_force
            phi = phi + self.timestep * dphi
            phi[mask2d <= 0] = -1.0

            change = float(np.linalg.norm(phi - last_phi) / (np.linalg.norm(last_phi) + 1e-8))
            energy = float(np.sum(e1 * memberships[0] + e2 * memberships[1] + e3 * memberships[2]))
            self.history["energy"].append(energy)
            self.history["change"].append(change)
            if self.verbose and (epoch == 1 or epoch % 50 == 0):
                print(f"Epoch/Iter {epoch:04d}/{self.epochs} | energy={energy:.6f} | change={change:.8f} | c={c} | ba={ba:.4f}")
            if change < self.tol:
                if self.verbose:
                    print(f"Stopping at epoch/iteration {epoch}; relative phi change < {self.tol}")
                break
            last_phi = phi.copy()

        seg = np.zeros_like(I, dtype=np.uint8)
        # Paper regions: Omega1 phi>k, Omega2 0<phi<=k, Omega3 phi<=0.
        # Mapping can be adjusted depending on intensity ordering. Default maps high/middle/low to WM/GM/CSF.
        seg[phi > self.k] = 2
        seg[(phi > 0) & (phi <= self.k)] = 1
        seg[phi <= 0] = 3
        seg[mask2d <= 0] = 0
        return seg, phi

    def fit_volume(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        seg = np.zeros_like(volume, dtype=np.uint8)
        for z in range(volume.shape[2]):
            if mask is not None and mask[:, :, z].sum() == 0:
                continue
            seg[:, :, z], _ = self.fit_slice(volume[:, :, z], None if mask is None else mask[:, :, z])
        return seg


# -----------------------------
# Deep baseline skeletons
# -----------------------------
if torch is not None:
    class DoubleConv2D(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class UNet2D(nn.Module):
        def __init__(self, in_channels=3, num_classes=4):
            super().__init__()
            self.d1 = DoubleConv2D(in_channels, 32)
            self.d2 = DoubleConv2D(32, 64)
            self.d3 = DoubleConv2D(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.c2 = DoubleConv2D(128, 64)
            self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.c1 = DoubleConv2D(64, 32)
            self.out = nn.Conv2d(32, num_classes, 1)
        def forward(self, x):
            x1 = self.d1(x); x2 = self.d2(self.pool(x1)); x3 = self.d3(self.pool(x2))
            y = self.u2(x3); y = self.c2(torch.cat([y, x2], dim=1))
            y = self.u1(y); y = self.c1(torch.cat([y, x1], dim=1))
            return self.out(y)

    class DoubleConv3D(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class UNet3D(nn.Module):
        def __init__(self, in_channels=3, num_classes=4):
            super().__init__()
            self.d1 = DoubleConv3D(in_channels, 16)
            self.d2 = DoubleConv3D(16, 32)
            self.d3 = DoubleConv3D(32, 64)
            self.pool = nn.MaxPool3d(2)
            self.u2 = nn.ConvTranspose3d(64, 32, 2, 2)
            self.c2 = DoubleConv3D(64, 32)
            self.u1 = nn.ConvTranspose3d(32, 16, 2, 2)
            self.c1 = DoubleConv3D(32, 16)
            self.out = nn.Conv3d(16, num_classes, 1)
        def forward(self, x):
            x1 = self.d1(x); x2 = self.d2(self.pool(x1)); x3 = self.d3(self.pool(x2))
            y = self.u2(x3); y = self.c2(torch.cat([y, x2], dim=1))
            y = self.u1(y); y = self.c1(torch.cat([y, x1], dim=1))
            return self.out(y)

    class MRBrainS2DSliceDataset(Dataset):
        def __init__(self, subject_dirs: List[Path], patch_size=(256, 224)):
            self.samples = []
            self.patch_size = patch_size
            for sd in subject_dirs:
                try:
                    t1 = load_nifti(locate_file(sd, ["t1"]))
                    tir = load_nifti(locate_file(sd, ["ir"]))
                    flair = load_nifti(locate_file(sd, ["flair"]))
                    lab = merge_mrbrains_labels(load_nifti(locate_file(sd, ["label"])))
                except Exception as e:
                    print(f"Skipping {sd}: {e}")
                    continue
                mask = lab > 0
                t1, tir, flair = zscore_normalize(t1, mask), zscore_normalize(tir, mask), zscore_normalize(flair, mask)
                for z in range(t1.shape[2]):
                    if mask[:, :, z].sum() > 20:
                        x = np.stack([t1[:, :, z], tir[:, :, z], flair[:, :, z]], axis=0)
                        y = lab[:, :, z]
                        self.samples.append((x.astype(np.float32), y.astype(np.int64)))
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            x, y = self.samples[i]
            x = torch.tensor(x).unsqueeze(0)
            y = torch.tensor(y).unsqueeze(0).float()
            x = F.interpolate(x, size=tuple(PAPER_CONFIG["deep_patch_size"]), mode="bilinear", align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=tuple(PAPER_CONFIG["deep_patch_size"]), mode="nearest").squeeze(0).squeeze(0).long()
            return x, y


def train_deep_baseline(model, train_loader, val_loader, epochs=1000, lr=1e-3, device="cuda"):
    if torch is None:
        raise ImportError("Install PyTorch to train deep baselines.")
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train(); tr_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); opt.step()
            tr_loss += loss.item()
        if ep == 1 or ep % 10 == 0:
            print(f"Epoch {ep:04d}/{epochs} | train_loss={tr_loss/max(1,len(train_loader)):.4f}")
    return model


def run_mab(args):
    data_root = Path(args.data_root)
    subjects = find_subjects(data_root)
    train_subjects = subjects[:PAPER_CONFIG["train_subjects"]]
    test_subjects = subjects[PAPER_CONFIG["train_subjects"]:PAPER_CONFIG["train_subjects"] + PAPER_CONFIG["test_subjects"]]
    if args.subject_index is not None:
        test_subjects = [subjects[args.subject_index]]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = MABSFDLS(
        epochs=args.epochs, sigma=args.sigma, mu=args.mu, nu=args.nu, timestep=args.timestep,
        k_level=args.k_level, eps=args.eps, tol=args.tol, verbose=True,
    )
    all_results = {}
    for sd in test_subjects:
        print(f"\nProcessing subject: {sd.name}")
        t1_path = locate_file(sd, ["t1"])
        img = load_nifti(t1_path)
        try:
            mask = load_nifti(locate_file(sd, ["brain", "mask"])) > 0
        except Exception:
            mask = img > np.percentile(img, 5)
        seg = model.fit_volume(img, mask=mask)
        out_path = out_dir / f"{sd.name}_MAB_SFDLS_seg.nii.gz"
        save_nifti_like(seg, t1_path, out_path)
        print(f"Saved segmentation: {out_path}")
        try:
            gt = merge_mrbrains_labels(load_nifti(locate_file(sd, ["label"])))
            all_results[sd.name] = evaluate_multiclass(seg, gt)
            print(json.dumps(all_results[sd.name], indent=2))
        except Exception as e:
            print(f"No label evaluation for {sd.name}: {e}")
    with open(out_dir / "mab_sfdls_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)


def run_deep(args, mode: str):
    if torch is None:
        raise ImportError("Install PyTorch to train deep baselines: pip install torch")
    subjects = find_subjects(Path(args.data_root))
    train_subjects = subjects[:PAPER_CONFIG["train_subjects"]]
    test_subjects = subjects[PAPER_CONFIG["train_subjects"]:PAPER_CONFIG["train_subjects"] + PAPER_CONFIG["test_subjects"]]
    train_ds = MRBrainS2DSliceDataset(train_subjects, patch_size=tuple(PAPER_CONFIG["deep_patch_size"]))
    val_ds = MRBrainS2DSliceDataset(test_subjects, patch_size=tuple(PAPER_CONFIG["deep_patch_size"]))
    train_loader = DataLoader(train_ds, batch_size=PAPER_CONFIG["deep_batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=PAPER_CONFIG["deep_batch_size"], shuffle=False)
    if mode == "unet2d":
        model = UNet2D(in_channels=3, num_classes=4)
    elif mode == "unet3d":
        raise NotImplementedError("3D training requires volumetric patch dataset; model class is provided as UNet3D.")
    else:
        raise NotImplementedError(
            f"{mode} should be run through its official implementation. The manuscript names this baseline but does not disclose source-level architecture/hyperparameters."
        )
    train_deep_baseline(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=args.device)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(Path(args.out_dir) / f"{mode}_mrbrains18.pt"))


def main():
    parser = argparse.ArgumentParser(description="MAB-SFDLS and MRBrainS18 segmentation framework")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs_mab_sfdls")
    parser.add_argument("--mode", type=str, default="mab", choices=["mab", "unet2d", "unet3d", "transunet", "nnunet", "umamba"])
    parser.add_argument("--epochs", type=int, default=PAPER_CONFIG["mab_epochs_reported"], help="Paper reports 1000 epochs/iterations for MAB-SFDLS convergence.")
    parser.add_argument("--subject_index", type=int, default=None)
    # MAB-SFDLS tunable values: not numerically disclosed in paper
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian kernel scale; paper denotes sigma but does not disclose numeric value.")
    parser.add_argument("--mu", type=float, default=0.2, help="Distance regularization weight; paper denotes mu but does not disclose numeric value.")
    parser.add_argument("--nu", type=float, default=0.003, help="Boundary smoothness weight; paper denotes nu but does not disclose numeric value.")
    parser.add_argument("--timestep", type=float, default=0.1, help="Euler time step; paper requires CFL stability but does not disclose numeric value.")
    parser.add_argument("--k_level", type=float, default=0.5, help="Dual threshold k; paper defines k-level-set but does not disclose numeric value.")
    parser.add_argument("--eps", type=float, default=1.0, help="Regularization epsilon for Heaviside/Dirac approximations.")
    parser.add_argument("--tol", type=float, default=1e-5)
    # Deep baseline training
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate not disclosed in paper; used only for local baseline skeleton.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Paper configuration:")
    print(json.dumps(PAPER_CONFIG, indent=2))

    if args.mode == "mab":
        run_mab(args)
    else:
        run_deep(args, args.mode)


if __name__ == "__main__":
    main()
