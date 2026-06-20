
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
