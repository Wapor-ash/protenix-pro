# RNA-specific loss weight overrides for Protenix fine-tuning.
#
# Based on analysis in claude_loss_an.md (Section 10.1):
#   - alpha_distogram: 0.03 -> 0.10 (enhance base-pairing distance learning)
#   - alpha_bond: 0.0 -> 0.5 (enable backbone bond constraints, AF3 fine-tuning standard)
#   - weight_rna: 5.0 -> 8.0 (increase RNA atom weight for RNA-only datasets)
#   - weight_smooth_lddt: keep 1.0 (actual weight already 4.0, sufficient)
#
# Usage:
#   Enable via config: --rna_loss.enable true
#   Override individual values: --rna_loss.alpha_distogram 0.15

import logging

logger = logging.getLogger(__name__)

# Default RNA loss overrides (recommended conservative values from analysis)
RNA_LOSS_DEFAULTS = {
    "alpha_distogram": 0.10,   # up from 0.03, ~3x increase
    "alpha_bond": 0.5,         # up from 0.0, actual weight = 4.0 * 0.5 = 2.0
    "weight_rna": 8.0,         # up from 5.0, for RNA-only datasets
}


def apply_rna_loss_overrides(loss_module, rna_loss_config):
    """
    Apply RNA-specific loss weight overrides to a ProtenixLoss module.

    This modifies loss_module.loss_weight dict and re-initializes MSELoss
    with the updated weight_rna if changed.

    Args:
        loss_module: ProtenixLoss instance (already initialized with default weights)
        rna_loss_config: config namespace with rna_loss parameters
    """
    from protenix.model.loss import MSELoss

    alpha_diffusion = loss_module.alpha_diffusion

    # --- Override alpha_distogram ---
    new_alpha_distogram = getattr(rna_loss_config, "alpha_distogram", RNA_LOSS_DEFAULTS["alpha_distogram"])
    old_distogram_w = loss_module.loss_weight["distogram_loss"]
    loss_module.loss_weight["distogram_loss"] = new_alpha_distogram
    logger.info(
        f"[RNA Loss] distogram_loss weight: {old_distogram_w} -> {new_alpha_distogram}"
    )

    # --- Override alpha_bond ---
    new_alpha_bond = getattr(rna_loss_config, "alpha_bond", RNA_LOSS_DEFAULTS["alpha_bond"])
    old_bond_w = loss_module.loss_weight["bond_loss"]
    new_bond_w = alpha_diffusion * new_alpha_bond
    loss_module.loss_weight["bond_loss"] = new_bond_w
    logger.info(
        f"[RNA Loss] bond_loss weight: {old_bond_w} -> {new_bond_w} "
        f"(alpha_diffusion={alpha_diffusion} * alpha_bond={new_alpha_bond})"
    )

    # --- Override weight_rna in MSELoss ---
    new_weight_rna = getattr(rna_loss_config, "weight_rna", RNA_LOSS_DEFAULTS["weight_rna"])
    old_weight_rna = loss_module.mse_loss.weight_rna
    if new_weight_rna != old_weight_rna:
        loss_module.mse_loss.weight_rna = new_weight_rna
        logger.info(
            f"[RNA Loss] MSELoss.weight_rna: {old_weight_rna} -> {new_weight_rna}"
        )

    # --- Log final loss weight summary ---
    logger.info("[RNA Loss] Final loss_weight dict:")
    for name, w in loss_module.loss_weight.items():
        logger.info(f"  {name}: {w}")
