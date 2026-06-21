
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-7


class DiceLoss(nn.Module):
    """Binary Dice loss."""

    def __init__(self, useSigmoid: bool = True, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.useSigmoid = useSigmoid
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.useSigmoid:
            input = torch.sigmoid(input)

        input = torch.flatten(input)
        target = torch.flatten(target).float()

        intersection = (input * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            input.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Binary Dice + BCE loss."""

    def __init__(self, useSigmoid: bool = True, smooth: float = 1.0):
        super(DiceBCELoss, self).__init__()
        self.useSigmoid = useSigmoid
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.useSigmoid:
            prob = torch.sigmoid(input)
        else:
            prob = input

        prob_flat = torch.flatten(prob)
        target_flat = torch.flatten(target).float()

        intersection = (prob_flat * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            prob_flat.sum() + target_flat.sum() + self.smooth
        )
        bce_loss = F.binary_cross_entropy(prob_flat, target_flat, reduction="mean")
        return bce_loss + dice_loss


class BCELoss(nn.Module):
    """Binary cross entropy loss."""

    def __init__(self, useSigmoid: bool = True):
        super(BCELoss, self).__init__()
        self.useSigmoid = useSigmoid

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.useSigmoid:
            input = torch.sigmoid(input)
        input = torch.flatten(input)
        target = torch.flatten(target).float()
        return F.binary_cross_entropy(input, target, reduction="mean")


class FocalLoss(nn.Module):
    """Binary focal loss."""

    def __init__(self, gamma: float = 2.0, eps: float = EPS):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(input)
        prob = torch.flatten(prob).clamp(self.eps, 1.0 - self.eps)
        target = torch.flatten(target).float()

        bce = F.binary_cross_entropy(prob, target, reduction="none")
        pt = torch.where(target == 1, prob, 1.0 - prob)
        focal = (1.0 - pt) ** self.gamma * bce
        return focal.mean()


class DiceFocalLoss(nn.Module):
    """Binary Dice + Focal loss."""

    def __init__(self, gamma: float = 2.0, smooth: float = 1.0, eps: float = EPS):
        super(DiceFocalLoss, self).__init__()
        self.dice = DiceLoss(useSigmoid=True, smooth=smooth)
        self.focal = FocalLoss(gamma=gamma, eps=eps)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice(input, target) + self.focal(input, target)


class MultiClassDiceLoss(nn.Module):
    """
    Multi-class Dice loss for MRBrainS18 tissue segmentation.

    Expected classes can be:
        0 = background
        1 = gray matter / GM
        2 = white matter / WM
        3 = cerebrospinal fluid / CSF

    input shape:  [B, C, H, W] or [B, C, D, H, W]
    target shape: [B, H, W] or [B, D, H, W] with class indices,
                  or one-hot [B, C, H, W] / [B, C, D, H, W]
    """

    def __init__(
        self,
        num_classes: int = 4,
        include_background: bool = False,
        smooth: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(input, dim=1)

        if target.dim() == input.dim() - 1:
            target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
            target_one_hot = target_one_hot.permute(0, -1, *range(1, target_one_hot.dim() - 1)).float()
        else:
            target_one_hot = target.float()

        start_class = 0 if self.include_background else 1
        dice_losses = []
        weights = []

        for cls in range(start_class, self.num_classes):
            pred_c = prob[:, cls].contiguous().view(prob.size(0), -1)
            target_c = target_one_hot[:, cls].contiguous().view(target_one_hot.size(0), -1)
            intersection = (pred_c * target_c).sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (
                pred_c.sum(dim=1) + target_c.sum(dim=1) + self.smooth
            )
            dice_losses.append(1.0 - dice.mean())
            if self.class_weights is not None:
                weights.append(self.class_weights[cls])

        losses = torch.stack(dice_losses)
        if self.class_weights is not None:
            weights_t = torch.stack(weights).to(losses.device)
            return (losses * weights_t).sum() / weights_t.sum().clamp_min(EPS)
        return losses.mean()


class BoundarySmoothnessLoss(nn.Module):
    """
    Smoothness regularization equivalent to the contour/boundary regularization
    term used in variational level-set segmentation.
    """

    def __init__(self):
        super(BoundarySmoothnessLoss, self).__init__()

    def forward(self, phi_or_prob: torch.Tensor) -> torch.Tensor:
        if phi_or_prob.dim() == 4:
            dx = torch.abs(phi_or_prob[:, :, :, 1:] - phi_or_prob[:, :, :, :-1]).mean()
            dy = torch.abs(phi_or_prob[:, :, 1:, :] - phi_or_prob[:, :, :-1, :]).mean()
            return dx + dy

        if phi_or_prob.dim() == 5:
            dx = torch.abs(phi_or_prob[:, :, :, :, 1:] - phi_or_prob[:, :, :, :, :-1]).mean()
            dy = torch.abs(phi_or_prob[:, :, :, 1:, :] - phi_or_prob[:, :, :, :-1, :]).mean()
            dz = torch.abs(phi_or_prob[:, :, 1:, :, :] - phi_or_prob[:, :, :-1, :, :]).mean()
            return dx + dy + dz

        raise ValueError("Expected 4-D or 5-D tensor.")


class LevelSetDistanceRegularization(nn.Module):
    """
    Distance regularization term based on p(s)=0.5*(s-1)^2 from the paper.
    It encourages |grad(phi)| close to 1 and reduces the need for reinitialization.
    """

    def __init__(self):
        super(LevelSetDistanceRegularization, self).__init__()

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.dim() == 4:
            gx = phi[:, :, :, 1:] - phi[:, :, :, :-1]
            gy = phi[:, :, 1:, :] - phi[:, :, :-1, :]
            gx = F.pad(gx, (0, 1, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1))
            grad_norm = torch.sqrt(gx.pow(2) + gy.pow(2) + EPS)
            return 0.5 * (grad_norm - 1.0).pow(2).mean()

        if phi.dim() == 5:
            gx = phi[:, :, :, :, 1:] - phi[:, :, :, :, :-1]
            gy = phi[:, :, :, 1:, :] - phi[:, :, :, :-1, :]
            gz = phi[:, :, 1:, :, :] - phi[:, :, :-1, :, :]
            gx = F.pad(gx, (0, 1, 0, 0, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1, 0, 0))
            gz = F.pad(gz, (0, 0, 0, 0, 0, 1))
            grad_norm = torch.sqrt(gx.pow(2) + gy.pow(2) + gz.pow(2) + EPS)
            return 0.5 * (grad_norm - 1.0).pow(2).mean()

        raise ValueError("Expected 4-D or 5-D level-set tensor.")


class MABSFDLSVariationalLoss(nn.Module):
    """
    Multiplicative-Additive Bias Single-Function Dual-Level-Set loss.

    This loss implements a differentiable PyTorch version of the paper's
    localized data-fidelity idea:
        |I(x) - b_m(x)c_i - b_a|^2
    combined with Dice supervision and variational regularization.

    Inputs:
        logits: network segmentation logits [B, C, H, W] or [B, C, D, H, W]
        image:  MRI image volume/slice [B, 1, H, W] or [B, 1, D, H, W]
        target: class mask indices or one-hot mask
        bm: multiplicative bias field, optional. If None, ones are used.
        ba: additive bias field/global bias, optional. If None, zeros are used.
        phi: level-set output, optional. If None, foreground probability is used.
    """

    def __init__(
        self,
        num_classes: int = 4,
        lambda_dice: float = 1.0,
        lambda_data: float = 1.0,
        lambda_boundary: float = 0.01,
        lambda_distance: float = 0.001,
        include_background: bool = False,
    ):
        super(MABSFDLSVariationalLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.lambda_data = lambda_data
        self.lambda_boundary = lambda_boundary
        self.lambda_distance = lambda_distance
        self.dice_loss = MultiClassDiceLoss(
            num_classes=num_classes,
            include_background=include_background,
        )
        self.boundary_loss = BoundarySmoothnessLoss()
        self.distance_reg = LevelSetDistanceRegularization()

    def _make_target_one_hot(self, target: torch.Tensor, input_dim: int) -> torch.Tensor:
        if target.dim() == input_dim - 1:
            target_one_hot = F.one_hot(target.long(), num_classes=self.num_classes)
            target_one_hot = target_one_hot.permute(0, -1, *range(1, target_one_hot.dim() - 1)).float()
            return target_one_hot
        return target.float()

    def forward(
        self,
        logits: torch.Tensor,
        image: torch.Tensor,
        target: torch.Tensor,
        bm: Optional[torch.Tensor] = None,
        ba: Optional[torch.Tensor] = None,
        phi: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        prob = F.softmax(logits, dim=1)
        target_one_hot = self._make_target_one_hot(target, logits.dim()).to(logits.device)

        if bm is None:
            bm = torch.ones_like(image)
        if ba is None:
            ba = torch.zeros_like(image)

        # Estimate region constants c_i from image and soft memberships.
        data_loss = torch.tensor(0.0, device=logits.device)
        for cls in range(self.num_classes):
            membership = prob[:, cls : cls + 1]
            denominator = membership.sum(dim=tuple(range(2, membership.dim())), keepdim=True).clamp_min(EPS)
            ci = (membership * image).sum(dim=tuple(range(2, image.dim())), keepdim=True) / denominator
            reconstructed = bm * ci + ba
            error = (image - reconstructed).pow(2)
            data_loss = data_loss + (membership * error).mean()

        dice = self.dice_loss(logits, target_one_hot)

        if phi is None:
            # Use foreground probability as a differentiable level-set surrogate.
            phi = prob[:, 1:].sum(dim=1, keepdim=True) if self.num_classes > 1 else prob

        boundary = self.boundary_loss(phi)
        distance = self.distance_reg(phi)

        total = (
            self.lambda_dice * dice
            + self.lambda_data * data_loss
            + self.lambda_boundary * boundary
            + self.lambda_distance * distance
        )

        components = {
            "total_loss": total.detach(),
            "dice_loss": dice.detach(),
            "mab_data_loss": data_loss.detach(),
            "boundary_loss": boundary.detach(),
            "distance_regularization": distance.detach(),
        }
        return total, components


class HybridLoss(nn.Module):
    """
    General hybrid loss used for baseline deep-learning segmentation models.
    Combines multi-class Dice and cross entropy.
    """

    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.5,
        include_background: bool = False,
    ):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.dice = MultiClassDiceLoss(num_classes=num_classes, include_background=include_background)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == input.dim():
            target_index = torch.argmax(target, dim=1).long()
        else:
            target_index = target.long()
        return self.alpha * self.ce(input, target_index) + (1.0 - self.alpha) * self.dice(input, target)

# -----------------------------------------------------------------------------
# Plotting utilities for metric visualization
# -----------------------------------------------------------------------------

def get_segmentation_metric_data() -> dict:
    """Return all metric values used by the plotting functions."""
    methods = ["U-Net", "3D U-Net", "TransUNet", "nnU-Net", "U-Mamba", "Proposed", "Original"]
    return {
        "methods": methods,
        "Dice": {
            "gray": [0.81, 0.83, 0.85, 0.87, 0.89, 0.86, 0.74],
            "white": [0.88, 0.90, 0.92, 0.88, 0.92, 0.95, 0.91],
            "ylim_min": 0.70,
        },
        "IoU": {
            "gray": [0.68, 0.71, 0.74, 0.78, 0.83, 0.74, 0.63],
            "white": [0.79, 0.82, 0.85, 0.78, 0.85, 0.88, 0.83],
            "ylim_min": 0.60,
        },
        "Accuracy": {
            "gray": [0.97, 0.98, 0.98, 0.99, 0.99, 0.96, 0.94],
            "white": [0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.98],
            "ylim_min": 0.70,
        },
        "Sensitivity": {
            "gray": [0.81, 0.83, 0.85, 0.85, 0.91, 0.84, 0.79],
            "white": [0.89, 0.91, 0.94, 0.87, 0.92, 0.97, 0.98],
            "ylim_min": 0.75,
        },
        "Specificity": {
            "gray": [0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.97],
            "white": [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.98],
            "ylim_min": 0.95,
        },
        "HD95 Gray": [3.90, 3.10, 2.20, 1.59, 1.35, 2.93, 20.45],
        "HD95 White": [3.25, 2.85, 2.50, 3.60, 2.40, 2.20, 3.10],
    }


def set_metric_plot_style(font_size: int = 18) -> None:
    """Apply a consistent matplotlib style for metric plots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size + 2,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    })


def bar3d_fake(
    ax,
    x: float,
    h: float,
    width: float = 0.35,
    depth: float = 0.06,
    face: str = "#1f77b4",
    side=None,
    top=None,
    shadow: bool = True,
    z: int = 3,
):
    """Draw a fake 3D bar at position x with height h."""
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe
    import numpy as np
    from matplotlib.patches import Polygon, Rectangle

    rgb = np.array(mcolors.to_rgb(face))
    if side is None:
        side = tuple(np.clip(rgb * 0.65, 0, 1))
    if top is None:
        top = tuple(np.clip(rgb * 1.10, 0, 1))

    dx = depth
    dy = depth * 0.55
    left = x - width / 2
    right = x + width / 2

    front = Rectangle((left, 0), width, h, facecolor=face, edgecolor="none", zorder=z)
    if shadow:
        front.set_path_effects([
            pe.SimplePatchShadow(offset=(3, -3), alpha=0.25, rho=0.98),
            pe.Normal(),
        ])
    ax.add_patch(front)

    ax.add_patch(Polygon(
        [(right, 0), (right + dx, dy), (right + dx, h + dy), (right, h)],
        closed=True,
        facecolor=side,
        edgecolor="none",
        zorder=z - 0.2,
    ))
    ax.add_patch(Polygon(
        [(left, h), (right, h), (right + dx, h + dy), (left + dx, h + dy)],
        closed=True,
        facecolor=top,
        edgecolor="none",
        zorder=z + 0.2,
    ))
    return front


def plot_grouped_3d_metric(
    metric_name: str,
    gray_values: list,
    white_values: list,
    methods: list,
    ylim_min: float,
    show: bool = True,
):
    """Plot grouped fake-3D bars for gray and white matter metric values."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    set_metric_plot_style(font_size=18)
    x = np.arange(len(methods))
    width = 0.35
    gap = 0.07
    depth = 0.04
    gray_color = "#1aa6b7"
    white_color = "#2f6fb2"

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_facecolor("#f6f7fb")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)

    for i, (gray_value, white_value) in enumerate(zip(gray_values, white_values)):
        x1 = x[i] - (width / 2 + gap / 2)
        x2 = x[i] + (width / 2 + gap / 2)
        bar3d_fake(ax, x1, gray_value, width=width, depth=depth, face=gray_color, shadow=True)
        bar3d_fake(ax, x2, white_value, width=width, depth=depth, face=white_color, shadow=True)

    dx = depth
    dy = depth * 0.55
    y_max = max(max(gray_values), max(white_values))
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_xlim(-0.5, len(methods) - 0.5 + dx + 0.15)
    ax.set_ylim(ylim_min, y_max + dy + 0.03)
    ax.margins(x=0.02)

    proxy1 = Rectangle((0, 0), 1, 1, fc=gray_color, ec="none")
    proxy2 = Rectangle((0, 0), 1, 1, fc=white_color, ec="none")
    ax.legend(
        [proxy1, proxy2],
        ["Gray Matter", "White Matter"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        frameon=False,
    )
    if show:
        plt.show()
    return fig, ax


def plot_all_grouped_3d_metrics(show: bool = True) -> list:
    """Plot Dice, IoU, Accuracy, Sensitivity, and Specificity charts."""
    data = get_segmentation_metric_data()
    figures = []
    for metric_name in ["Dice", "IoU", "Accuracy", "Sensitivity", "Specificity"]:
        metric_data = data[metric_name]
        figures.append(plot_grouped_3d_metric(
            metric_name=metric_name,
            gray_values=metric_data["gray"],
            white_values=metric_data["white"],
            methods=data["methods"],
            ylim_min=metric_data["ylim_min"],
            show=show,
        ))
    return figures


def plot_hd95(values: list, methods: list, tissue_name: str, show: bool = True):
    """Plot an HD95 bar chart for one tissue type."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })
    fig, ax = plt.subplots()
    ax.bar(methods, values)
    ax.set_ylabel("HD95 (mm)")
    ax.set_title(f"HD95 – {tissue_name} Matter (Lower is Better)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_hd95_charts(show: bool = True) -> list:
    """Plot gray-matter and white-matter HD95 charts."""
    data = get_segmentation_metric_data()
    return [
        plot_hd95(data["HD95 Gray"], data["methods"], "Gray", show=show),
        plot_hd95(data["HD95 White"], data["methods"], "White", show=show),
    ]


def plot_level_set_surface(show: bool = True):
    """Create and plot a brain-like signed distance level-set surface."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    from skimage.draw import polygon

    height, width = 240, 240
    cx, cy = width // 2, height // 2
    theta = np.linspace(0, 2 * np.pi, 900, endpoint=False)
    radius = 70 + 10 * np.sin(6 * theta) + 6 * np.sin(13 * theta + 0.7)
    x_boundary = cx + radius * np.cos(theta)
    y_boundary = cy + 0.85 * radius * np.sin(theta)

    rr, cc = polygon(y_boundary, x_boundary, shape=(height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rr, cc] = 1

    outside = distance_transform_edt(1 - mask)
    inside = distance_transform_edt(mask)
    phi = gaussian_filter(outside - inside, sigma=2.0)
    phi = phi / np.max(np.abs(phi)) * 65
    y_grid, x_grid = np.mgrid[0:height, 0:width]

    fig = plt.figure(figsize=(5.4, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, phi, cmap="jet", rstride=1, cstride=1, linewidth=0, antialiased=True, edgecolor="none")
    ax.view_init(elev=25, azim=-135)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(-70, 70)
    ax.grid(True)
    ax.set_xlabel("Spatial Coordinate X")
    ax.set_ylabel("Spatial Coordinate Y")
    ax.set_zlabel("Level Set Function ϕ(x, y)", labelpad=15)
    if show:
        plt.show()
    return fig, ax


def plot_radar_chart(tissue: str = "white", show: bool = True):
    """Plot a radar chart for gray or white matter metrics."""
    import matplotlib.pyplot as plt
    import numpy as np

    tissue = tissue.lower()
    data = get_segmentation_metric_data()
    methods = data["methods"]
    metrics = ["Dice", "IoU", "Accuracy", "Sensitivity", "Specificity"]
    metric_values = {
        method: [data[metric][tissue][i] for metric in metrics]
        for i, method in enumerate(methods)
    }
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams.update({
        "figure.dpi": 140,
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.facecolor": "#fbfbfd",
        "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(True, linewidth=0.8, alpha=0.35)
    ax.spines["polar"].set_alpha(0.25)
    ax.set_ylim(0.60, 1.00)
    rticks = [0.70, 0.80, 0.90, 1.00]
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{tick:.2f}" for tick in rticks], alpha=0.7)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        label.set_alpha(0.9)

    colors = plt.cm.tab10.colors
    for i, method in enumerate(methods):
        values = metric_values[method] + metric_values[method][:1]
        ax.plot(angles, values, color=colors[i % 10], linewidth=2.4, alpha=0.95, label=method)
        ax.fill(angles, values, color=colors[i % 10], alpha=0.08)

    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=True, borderpad=0.7, labelspacing=0.55)
    leg.get_frame().set_alpha(0.92)
    leg.get_frame().set_edgecolor((0, 0, 0, 0.12))
    ax.set_title(f"{tissue.title()} Matter Metrics")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_all_metric_figures(show: bool = True) -> dict:
    """Call all plotting functions and return their figures/axes."""
    return {
        "grouped_3d_metrics": plot_all_grouped_3d_metrics(show=show),
        "hd95": plot_hd95_charts(show=show),
        "level_set_surface": plot_level_set_surface(show=show),
        "white_radar": plot_radar_chart(tissue="white", show=show),
        "gray_radar": plot_radar_chart(tissue="gray", show=show),
    }


# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    num_classes = 4  # background, GM, WM, CSF
    h, w = 128, 128

    logits = torch.randn(batch_size, num_classes, h, w).to(device)
    image = torch.randn(batch_size, 1, h, w).to(device)
    target = torch.randint(0, num_classes, (batch_size, h, w)).to(device)

    loss_function = MABSFDLSVariationalLoss(
        num_classes=num_classes,
        lambda_dice=1.0,
        lambda_data=1.0,
        lambda_boundary=0.01,
        lambda_distance=0.001,
    ).to(device)

    loss, loss_items = loss_function(logits=logits, image=image, target=target)
    print("MAB-SFDLS loss:", float(loss.item()))
    print({k: float(v.item()) for k, v in loss_items.items()})

    # Call plotting functions. Set show=True to display all figures when running this file.
    plot_all_metric_figures(show=True)
