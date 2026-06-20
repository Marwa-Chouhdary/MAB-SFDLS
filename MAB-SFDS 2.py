
import os
import glob
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_erosion
except ImportError:
    gaussian_filter = None
    distance_transform_edt = None
    binary_erosion = None


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Dataset: MRBrainS18 NIfTI Loader
# -----------------------------
class Dataset_MRBrainS18_2D(Dataset):
    """
    Slice-wise MRBrainS18 loader.

    Expected subject folders may contain files such as:
        *T1*.nii.gz, *T1_IR*.nii.gz, *T2_FLAIR*.nii.gz, *label*.nii.gz

    This loader is intentionally flexible. You may also provide explicit file names
    by editing _find_file().
    """

    def __init__(
        self,
        root_dir: str,
        subject_ids: Optional[List[str]] = None,
        slice_axis: int = 2,
        normalize: str = "zscore",
        use_modalities: Tuple[str, ...] = ("T1", "T1_IR", "T2_FLAIR"),
        label_map: Optional[Dict[int, int]] = None,
        brain_mask_nonzero: bool = True,
    ):
        if nib is None:
            raise ImportError("Please install nibabel: pip install nibabel")

        self.root_dir = root_dir
        self.slice_axis = slice_axis
        self.normalize = normalize
        self.use_modalities = use_modalities
        self.brain_mask_nonzero = brain_mask_nonzero

        # Output labels: 0 background/ignored, 1 GM, 2 WM, 3 CSF
        # Adjust these IDs according to the exact MRBrainS18 label encoding.
        self.label_map = label_map or {
            1: 1,  # cortical gray matter -> GM
            2: 1,  # basal ganglia -> GM
            3: 2,  # white matter -> WM
            4: 2,  # white matter lesions -> WM
            5: 3,  # CSF -> CSF
            6: 3,  # ventricles -> CSF
        }

        all_subjects = sorted([p for p in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(p)])
        if subject_ids is not None:
            wanted = set(subject_ids)
            all_subjects = [p for p in all_subjects if os.path.basename(p) in wanted]
        if len(all_subjects) == 0:
            raise FileNotFoundError(f"No subject folders found in {root_dir}")

        self.samples = []
        for subject_path in all_subjects:
            try:
                vols, lab = self._load_subject(subject_path)
            except Exception as exc:
                print(f"Skipping {subject_path}: {exc}")
                continue

            n_slices = vols.shape[self.slice_axis + 1]  # vols shape: C,H,W,D usually after load
            for s in range(n_slices):
                if self._slice_has_brain(vols, lab, s):
                    self.samples.append((subject_path, s))

        if len(self.samples) == 0:
            raise RuntimeError("No valid slices found. Check paths and label files.")

    def _find_file(self, subject_path: str, key: str) -> str:
        patterns = {
            "T1": ["*T1*.nii.gz", "*t1*.nii.gz"],
            "T1_IR": ["*T1_IR*.nii.gz", "*T1-IR*.nii.gz", "*IR*.nii.gz", "*ir*.nii.gz"],
            "T2_FLAIR": ["*T2_FLAIR*.nii.gz", "*T2-FLAIR*.nii.gz", "*FLAIR*.nii.gz", "*flair*.nii.gz"],
            "label": ["*label*.nii.gz", "*seg*.nii.gz", "*mask*.nii.gz", "*Labels*.nii.gz"],
        }
        for pat in patterns[key]:
            found = sorted(glob.glob(os.path.join(subject_path, pat)))
            if found:
                return found[0]
        raise FileNotFoundError(f"Could not find {key} file in {subject_path}")

    def _normalize_volume(self, vol: np.ndarray) -> np.ndarray:
        vol = vol.astype(np.float32)
        mask = vol > 0 if self.brain_mask_nonzero else np.ones_like(vol, dtype=bool)
        if self.normalize == "zscore":
            mean = vol[mask].mean() if mask.any() else vol.mean()
            std = vol[mask].std() if mask.any() else vol.std()
            vol = (vol - mean) / (std + 1e-8)
        elif self.normalize == "minmax":
            lo, hi = np.percentile(vol[mask], [1, 99]) if mask.any() else np.percentile(vol, [1, 99])
            vol = np.clip((vol - lo) / (hi - lo + 1e-8), 0, 1)
        return vol.astype(np.float32)

    def _load_subject(self, subject_path: str):
        vols = []
        for mod in self.use_modalities:
            f = self._find_file(subject_path, mod)
            vol = nib.load(f).get_fdata()
            vols.append(self._normalize_volume(vol))
        label_file = self._find_file(subject_path, "label")
        lab_raw = nib.load(label_file).get_fdata().astype(np.int16)
        lab = np.zeros_like(lab_raw, dtype=np.int64)
        for src, dst in self.label_map.items():
            lab[lab_raw == src] = dst
        vols = np.stack(vols, axis=0)  # C,H,W,D
        return vols, lab

    def _get_slice(self, arr: np.ndarray, s: int):
        if arr.ndim == 4:
            if self.slice_axis == 0:
                return arr[:, s, :, :]
            if self.slice_axis == 1:
                return arr[:, :, s, :]
            return arr[:, :, :, s]
        if self.slice_axis == 0:
            return arr[s, :, :]
        if self.slice_axis == 1:
            return arr[:, s, :]
        return arr[:, :, s]

    def _slice_has_brain(self, vols, lab, s):
        img = self._get_slice(vols, s)
        mask = self._get_slice(lab, s)
        return (np.abs(img).sum() > 1e-6) and ((mask > 0).sum() > 10)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_path, s = self.samples[idx]
        vols, lab = self._load_subject(subject_path)
        img = self._get_slice(vols, s).astype(np.float32)
        mask = self._get_slice(lab, s).astype(np.int64)
        return torch.from_numpy(img), torch.from_numpy(mask)


# -----------------------------
# MAB-SFDLS Variational Model
# -----------------------------
def heaviside_eps(phi: np.ndarray, eps: float = 1.0) -> np.ndarray:
    return 0.5 * (1.0 + (2.0 / np.pi) * np.arctan(phi / eps))


def dirac_eps(phi: np.ndarray, eps: float = 1.0) -> np.ndarray:
    return (eps / np.pi) / (eps * eps + phi * phi)


def gradient_norm(phi: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(phi)
    return np.sqrt(gx * gx + gy * gy + 1e-8)


def curvature(phi: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(phi)
    norm = np.sqrt(gx * gx + gy * gy + 1e-8)
    nx, ny = gx / norm, gy / norm
    nxy = np.gradient(nx, axis=1)
    nyx = np.gradient(ny, axis=0)
    return nxy + nyx


def distance_regularization(phi: np.ndarray) -> np.ndarray:
    # Approximate div(d_p(|grad phi|) grad phi) for p(s)=0.5(s-1)^2
    gy, gx = np.gradient(phi)
    s = np.sqrt(gx * gx + gy * gy + 1e-8)
    dps = (s - 1.0) / (s + 1e-8)
    vx = dps * gx
    vy = dps * gy
    return np.gradient(vx, axis=1) + np.gradient(vy, axis=0)


def initialize_phi(image2d: np.ndarray, k: float) -> np.ndarray:
    """Initialize a single level-set function using intensity quantiles."""
    img = image2d.copy()
    img = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1) + 1e-8)
    img = np.clip(img, 0, 1)
    # Map intensities to a signed implicit function so that zero and k split three regions.
    phi = 2.0 * img - 0.5
    phi = gaussian_filter(phi, sigma=1.0) if gaussian_filter else phi
    return phi.astype(np.float32)


class MABSFDLS:
    """Multiplicative-Additive Bias Single-Function Dual-Level-Set optimizer."""

    def __init__(self, config: Dict):
        self.config = config
        if gaussian_filter is None:
            raise ImportError("Please install scipy: pip install scipy")

    def _memberships(self, phi: np.ndarray):
        k = self.config["k_threshold"]
        eps = self.config["epsilon"]
        h0 = heaviside_eps(phi, eps)
        hk = heaviside_eps(phi - k, eps)
        m1 = hk
        m2 = h0 - hk
        m3 = 1.0 - h0
        return [m1, m2, m3]

    def _update_region_constants(self, I, bm, ba, memberships):
        cs = []
        for m in memberships:
            num = np.sum(m * bm * (I - ba))
            den = np.sum(m * bm * bm) + 1e-8
            cs.append(float(num / den))
        return np.array(cs, dtype=np.float32)

    def _update_bias_fields(self, I, c, memberships):
        expected = np.zeros_like(I, dtype=np.float32)
        for ci, mi in zip(c, memberships):
            expected += ci * mi

        # Multiplicative smooth bias field bm(x)
        numerator = gaussian_filter(I * expected, sigma=self.config["sigma"])
        denominator = gaussian_filter(expected * expected, sigma=self.config["sigma"]) + 1e-8
        bm = numerator / denominator
        bm = gaussian_filter(bm, sigma=self.config["sigma"])
        bm = np.clip(bm, 0.2, 5.0).astype(np.float32)

        # Global additive bias ba
        residual = I - bm * expected
        ba = float(np.mean(residual))
        return bm, ba

    def _local_errors(self, I, bm, ba, c):
        errors = []
        for ci in c:
            e = (I - bm * ci - ba) ** 2
            e = gaussian_filter(e, sigma=self.config["sigma"])
            errors.append(e)
        return errors

    def segment_slice(self, image: np.ndarray):
        """
        Segment one 2-D slice into three tissue regions:
        1: GM-like, 2: WM-like, 3: CSF-like.
        """
        # Use first modality as level-set driving image, or averaged multimodal intensity.
        if image.ndim == 3:
            I = image.mean(axis=0)
        else:
            I = image
        I = I.astype(np.float32)
        I = (I - I.mean()) / (I.std() + 1e-8)

        phi = initialize_phi(I, self.config["k_threshold"])
        bm = np.ones_like(I, dtype=np.float32)
        ba = 0.0
        c = np.array([-0.5, 0.0, 0.5], dtype=np.float32)

        for it in range(self.config["iterations"]):
            memberships = self._memberships(phi)
            c = self._update_region_constants(I, bm, ba, memberships)
            bm, ba = self._update_bias_fields(I, c, memberships)
            e1, e2, e3 = self._local_errors(I, bm, ba, c)

            eps = self.config["epsilon"]
            k = self.config["k_threshold"]
            nu = self.config["nu"]
            mu = self.config["mu"]
            dt = self.config["time_step"]

            d0 = dirac_eps(phi, eps)
            dk = dirac_eps(phi - k, eps)

            # Gradient descent for Eq. 9 memberships:
            # M1=H(phi-k), M2=H(phi)-H(phi-k), M3=1-H(phi)
            data_force = -dk * (e1 - e2) - d0 * (e2 - e3)
            length_force = nu * curvature(phi) * (d0 + dk)
            reg_force = mu * distance_regularization(phi)
            phi = phi + dt * (data_force + length_force + reg_force)

            if it > 5 and it % self.config["check_interval"] == 0:
                # simple numerical stabilization
                phi = np.clip(phi, -5.0, 5.0)

        m1, m2, m3 = self._memberships(phi)
        seg = np.zeros_like(I, dtype=np.uint8)
        scores = np.stack([m1, m2, m3], axis=0)
        seg = np.argmax(scores, axis=0).astype(np.uint8) + 1
        seg[np.abs(I) < self.config["background_threshold"]] = 0
        return seg, phi, bm, ba, c


# -----------------------------
# Metrics
# -----------------------------
def binary_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)
    acc = (tp + tn + 1e-8) / (tp + tn + fp + fn + 1e-8)
    sen = (tp + 1e-8) / (tp + fn + 1e-8)
    spe = (tn + 1e-8) / (tn + fp + 1e-8)
    return {"Dice": float(dice), "IoU": float(iou), "Accuracy": float(acc), "Sensitivity": float(sen), "Specificity": float(spe)}


def hd95(pred: np.ndarray, target: np.ndarray, voxel_spacing=(0.958, 0.958)) -> float:
    if distance_transform_edt is None or binary_erosion is None:
        return float("nan")
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")
    pred_border = np.logical_xor(pred, binary_erosion(pred))
    target_border = np.logical_xor(target, binary_erosion(target))
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)
    dt_target = distance_transform_edt(~target_border, sampling=voxel_spacing)
    d1 = dt_target[pred_border]
    d2 = dt_pred[target_border]
    if d1.size == 0 or d2.size == 0:
        return float("nan")
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# -----------------------------
# Agent Class: same style as attached file
# -----------------------------
class Agent_MABSFDLS:
    def __init__(self, model: MABSFDLS, config: Dict):
        self.model = model
        self.config = config
        self.epoch = 0
        self.history = []

    def train(self, data_loader):
        """
        MAB-SFDLS has no neural weights. This function performs repeated variational
        fitting/evaluation in an epoch-style loop to match the attached Agent code style.
        """
        for epoch in range(self.config["n_epoch"]):
            running_dice = []
            for i, (inputs, labels) in enumerate(data_loader):
                # Process slice-wise on CPU numpy because this is variational optimization.
                inputs_np = inputs.numpy()
                labels_np = labels.numpy()
                batch_dice = []

                for b in range(inputs_np.shape[0]):
                    seg, phi, bm, ba, c = self.model.segment_slice(inputs_np[b])
                    # Average Dice over GM, WM, CSF labels.
                    cls_scores = []
                    for cls_id in self.config["classes"]:
                        cls_scores.append(binary_metrics(seg == cls_id, labels_np[b] == cls_id)["Dice"])
                    batch_dice.append(float(np.nanmean(cls_scores)))

                running_dice.append(float(np.nanmean(batch_dice)))

                if i % self.config["save_interval"] == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.config['n_epoch']}], "
                        f"Step [{i + 1}/{len(data_loader)}], "
                        f"Mean Dice: {running_dice[-1]:.4f}"
                    )

            avg_dice = float(np.nanmean(running_dice))
            self.history.append({"epoch": epoch + 1, "mean_dice": avg_dice})
            print(f"Epoch [{epoch + 1}/{self.config['n_epoch']}], Average Dice: {avg_dice:.4f}")

            if (epoch + 1) % self.config["evaluate_interval"] == 0:
                os.makedirs(self.config["model_path"], exist_ok=True)
                np.save(os.path.join(self.config["model_path"], f"history_epoch_{epoch+1}.npy"), self.history)

    def evaluate(self, data_loader):
        metrics_per_class = {cls: [] for cls in self.config["classes"]}
        hd_per_class = {cls: [] for cls in self.config["classes"]}

        for inputs, labels in data_loader:
            inputs_np = inputs.numpy()
            labels_np = labels.numpy()
            for b in range(inputs_np.shape[0]):
                seg, phi, bm, ba, c = self.model.segment_slice(inputs_np[b])
                for cls_id in self.config["classes"]:
                    met = binary_metrics(seg == cls_id, labels_np[b] == cls_id)
                    metrics_per_class[cls_id].append(met)
                    hd_per_class[cls_id].append(hd95(seg == cls_id, labels_np[b] == cls_id, self.config["voxel_spacing_2d"]))

        summary = {}
        for cls_id in self.config["classes"]:
            name = self.config["class_names"][cls_id]
            cls_list = metrics_per_class[cls_id]
            summary[name] = {
                key: float(np.nanmean([m[key] for m in cls_list]))
                for key in ["Dice", "IoU", "Accuracy", "Sensitivity", "Specificity"]
            }
            summary[name]["HD95"] = float(np.nanmean(hd_per_class[cls_id]))
        return summary


# -----------------------------
# Configuration
# -----------------------------
config = [{
    # Basic dataset paths
    "img_path": r"/content/drive/MyDrive/MRBrainS18",   # subject folders containing MRI + label NIfTI files
    "label_path": r"same_as_subject_folder",            # labels are searched inside each subject folder
    "model_path": r"Models/MAB_SFDLS_Run1",
    "device": "cpu",  # MAB-SFDLS implementation is numpy/scipy CPU-based
    "unlock_CPU": True,

    # Paper dataset details
    "dataset": "MRBrainS18",
    "modalities": ("T1", "T1_IR", "T2_FLAIR"),
    "num_subjects": 30,
    "train_subjects": 24,
    "test_subjects": 6,
    "classes": [1, 2, 3],
    "class_names": {1: "Gray Matter", 2: "White Matter", 3: "CSF"},
    "voxel_spacing_2d": (0.958, 0.958),

    # Training-style loop, following the attached agent style
    "save_interval": 10,
    "evaluate_interval": 10,
    "n_epoch": 1,  # variational model; set >1 only if you need repeated runs like the template
    "batch_size": 1,
    "data_split": [24, 6],

    # MAB-SFDLS optimization parameters
    # The paper names these as manually selected parameters but does not give fixed values.
    "iterations": 100,
    "time_step": 0.1,
    "check_interval": 10,
    "k_threshold": 0.5,
    "epsilon": 1.0,
    "sigma": 3.0,
    "mu": 0.2,
    "nu": 0.003,
    "background_threshold": -5.0,

    # Data settings
    "slice_axis": 2,
    "normalize": "zscore",
    "seed": 42,
}]


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = config[0]
    set_seed(cfg["seed"])

    dataset = Dataset_MRBrainS18_2D(
        root_dir=cfg["img_path"],
        slice_axis=cfg["slice_axis"],
        normalize=cfg["normalize"],
        use_modalities=cfg["modalities"],
    )

    # Subject-level split should be used in final experiments.
    # This code uses slice index split as a runnable fallback when subject IDs are not provided.
    n = len(dataset)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = MABSFDLS(cfg)
    agent = Agent_MABSFDLS(model, cfg)

    agent.train(train_loader)
    evaluation_metrics = agent.evaluate(test_loader)

    print("\nEvaluation Metrics:")
    for cls_name, vals in evaluation_metrics.items():
        print(cls_name, vals)

    os.makedirs(cfg["model_path"], exist_ok=True)
    np.save(os.path.join(cfg["model_path"], "final_metrics.npy"), evaluation_metrics)


if __name__ == "__main__":
    main()
