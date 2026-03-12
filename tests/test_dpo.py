"""
Tests for the DPO training module.

Covers the DPO loss function, log-probability computation,
collation, and dataset loading.
"""

import pytest
import torch

from neurel_deob.training.dpo_finetune import (
    dpo_loss,
    dpo_collate_fn,
    DPOConfig,
)


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

class TestDPOLoss:
    """Test the DPO loss function."""

    def test_equal_logps_gives_log2_loss(self):
        """When policy == ref for both chosen/rejected, loss ≈ log(2)."""
        chosen = torch.tensor([-1.0])
        rejected = torch.tensor([-2.0])
        loss, c_rew, r_rew = dpo_loss(
            chosen, rejected, chosen, rejected,
            beta=0.1,
        )
        # margin is 0, so loss = -log(σ(0)) = log(2) ≈ 0.6931
        assert abs(loss.item() - 0.6931) < 0.01

    def test_positive_margin_lowers_loss(self):
        """When policy prefers chosen more than ref does, loss < log(2)."""
        ref_c = torch.tensor([-1.0])
        ref_r = torch.tensor([-2.0])
        # Policy has higher chosen log-prob (better)
        pol_c = torch.tensor([-0.5])
        pol_r = torch.tensor([-2.5])
        loss, c_rew, r_rew = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        assert loss.item() < 0.6931
        assert c_rew.item() > r_rew.item()

    def test_negative_margin_raises_loss(self):
        """When policy prefers rejected, loss > log(2)."""
        ref_c = torch.tensor([-1.0])
        ref_r = torch.tensor([-2.0])
        # Policy prefers rejected (bad)
        pol_c = torch.tensor([-2.0])
        pol_r = torch.tensor([-0.5])
        loss, c_rew, r_rew = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        assert loss.item() > 0.6931

    def test_beta_scales_loss(self):
        """Higher beta should amplify the margin effect."""
        ref_c = torch.tensor([-1.0])
        ref_r = torch.tensor([-2.0])
        pol_c = torch.tensor([-0.5])
        pol_r = torch.tensor([-2.5])

        loss_low, _, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.01)
        loss_high, _, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=1.0)
        # With positive margin, higher beta → lower loss (more confident)
        assert loss_high.item() < loss_low.item()

    def test_ipo_loss_type(self):
        """IPO loss should compute without error."""
        chosen = torch.tensor([-1.0, -1.5])
        rejected = torch.tensor([-2.0, -3.0])
        ref_c = torch.tensor([-1.2, -1.6])
        ref_r = torch.tensor([-2.1, -3.1])
        loss, _, _ = dpo_loss(chosen, rejected, ref_c, ref_r,
                              beta=0.1, loss_type="ipo")
        assert loss.item() >= 0

    def test_batch_dimension(self):
        """Loss should handle batch dimension correctly."""
        B = 4
        loss, c, r = dpo_loss(
            torch.randn(B), torch.randn(B),
            torch.randn(B), torch.randn(B),
            beta=0.1,
        )
        assert loss.dim() == 0  # scalar
        assert c.dim() == 0
        assert r.dim() == 0

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="Unknown DPO loss"):
            dpo_loss(
                torch.tensor([0.0]), torch.tensor([0.0]),
                torch.tensor([0.0]), torch.tensor([0.0]),
                loss_type="invalid",
            )


# ---------------------------------------------------------------------------
# DPO collation
# ---------------------------------------------------------------------------

class TestDPOCollation:
    """Test the DPO collation function."""

    def test_collate_basic(self):
        batch = [
            {
                "prompt_ids": torch.tensor([1, 2, 3]),
                "chosen_ids": torch.tensor([4, 5]),
                "rejected_ids": torch.tensor([6, 7, 8]),
            },
            {
                "prompt_ids": torch.tensor([1, 2]),
                "chosen_ids": torch.tensor([4, 5, 6]),
                "rejected_ids": torch.tensor([7]),
            },
        ]
        result = dpo_collate_fn(batch, pad_id=0, label_pad_id=-100)
        assert result is not None
        assert result["prompt_ids"].shape[0] == 2
        assert result["chosen_labels"].shape[0] == 2
        assert result["rejected_labels"].shape[0] == 2
        # Prompts should be padded to same length
        assert result["prompt_ids"].shape[1] == 3  # max prompt len

    def test_collate_filters_none(self):
        batch = [None, {"prompt_ids": torch.tensor([1]),
                         "chosen_ids": torch.tensor([2]),
                         "rejected_ids": torch.tensor([3])}, None]
        result = dpo_collate_fn(batch)
        assert result is not None
        assert result["prompt_ids"].shape[0] == 1

    def test_collate_all_none(self):
        result = dpo_collate_fn([None, None])
        assert result is None

    def test_decoder_input_ids_right_shifted(self):
        batch = [{
            "prompt_ids": torch.tensor([10, 20]),
            "chosen_ids": torch.tensor([30, 40, 50]),
            "rejected_ids": torch.tensor([60, 70]),
        }]
        result = dpo_collate_fn(batch, pad_id=0, label_pad_id=-100)
        # decoder_input_ids[0] should be BOS (0)
        assert result["chosen_decoder_input_ids"][0, 0].item() == 0
        assert result["rejected_decoder_input_ids"][0, 0].item() == 0


# ---------------------------------------------------------------------------
# DPOConfig
# ---------------------------------------------------------------------------

class TestDPOConfig:
    """Test DPO configuration."""

    def test_defaults(self):
        cfg = DPOConfig()
        assert cfg.beta == 0.1
        assert cfg.lr == 5e-7
        assert cfg.loss_type == "sigmoid"

    def test_effective_batch_size(self):
        cfg = DPOConfig(batch_size=2, gradient_accumulation_steps=8)
        assert cfg.effective_batch_size == 16
