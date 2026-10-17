#!/usr/bin/env python3
"""Loss functions for appraisal-conditioned emotion (AE) training.

This module keeps the three objectives separate so the trainer can log and
schedule them independently:

* conditional emotion SFT on ``P(E | S, A)``;
* paired appraisal contrast over matched and crossed appraisal/emotion paths;
* full-chain SFT replay for preservation.

The first preservation implementation is replay only. A reference-model KL is
intentionally not mixed into this version because it has a different scale and
requires a separately validated teacher-forward path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


IGNORE_INDEX = -100


@dataclass(frozen=True)
class SequenceLogLikelihood:
    """Per-sequence mean log-likelihood over supervised causal-LM tokens."""

    mean_log_likelihood: Tensor
    token_log_likelihood_sum: Tensor
    token_count: Tensor


@dataclass(frozen=True)
class PairContrastiveResult:
    loss: Tensor
    margin_a: Tensor
    margin_b: Tensor
    accuracy_a: Tensor
    accuracy_b: Tensor
    per_pair_loss: Tensor


@dataclass(frozen=True)
class AELossOutput:
    """Weighted AE objective with separately accessible loss components."""

    total_loss: Tensor
    conditional_loss: Tensor
    pair_loss: Tensor
    preservation_loss: Tensor
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)

    def detached_metrics(self, prefix: str = "ae") -> dict[str, Tensor]:
        """Return detached tensors suitable for distributed-aware logging."""
        metrics = {
            f"{prefix}/loss_total": self.total_loss.detach(),
            f"{prefix}/loss_conditional": self.conditional_loss.detach(),
            f"{prefix}/loss_pair": self.pair_loss.detach(),
            f"{prefix}/loss_preservation": self.preservation_loss.detach(),
        }
        metrics.update(
            {
                f"{prefix}/{name}": value.detach()
                for name, value in self.diagnostics.items()
            }
        )
        return metrics


def _require_logits_and_labels(logits: Tensor, labels: Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [batch, sequence, vocabulary], got "
            f"{tuple(logits.shape)}"
        )
    if labels.ndim != 2:
        raise ValueError(
            f"labels must have shape [batch, sequence], got {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels batch/sequence dimensions differ: "
            f"{tuple(logits.shape[:2])} != {tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("At least two sequence positions are required")


def extract_logits(model_output: Any) -> Tensor:
    """Extract logits from a Transformers output, mapping, or logits tensor."""
    if isinstance(model_output, Tensor):
        return model_output
    logits = getattr(model_output, "logits", None)
    if logits is None and isinstance(model_output, Mapping):
        logits = model_output.get("logits")
    if not isinstance(logits, Tensor):
        raise TypeError("Model output does not contain tensor logits")
    return logits


def sequence_mean_log_likelihood(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> SequenceLogLikelihood:
    """Compute causal mean token log-likelihood for every sequence.

    The one-token causal shift is applied here. Every sequence must contain at
    least one supervised token after the shift; silently accepting an empty
    target would make both conditional and contrastive losses invalid.
    """
    _require_logits_and_labels(logits, labels)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(ignore_index)
    token_count = valid.sum(dim=1)
    if bool(token_count.eq(0).any().item()):
        indices = token_count.eq(0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "Every sequence needs a supervised target after causal shift; "
            f"empty rows={indices[:10]}"
        )

    token_nll = F.cross_entropy(
        shift_logits.float().reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape_as(shift_labels)
    token_log_likelihood = -token_nll.masked_fill(~valid, 0.0)
    token_sum = token_log_likelihood.sum(dim=1)
    mean_log_likelihood = token_sum / token_count.to(token_sum.dtype)
    return SequenceLogLikelihood(
        mean_log_likelihood=mean_log_likelihood,
        token_log_likelihood_sum=token_sum,
        token_count=token_count,
    )


def _weighted_mean(values: Tensor, weights: Tensor | None, name: str) -> Tensor:
    if values.ndim != 1:
        raise ValueError(f"{name} values must be one-dimensional")
    if weights is None:
        return values.mean()
    weights = weights.detach().to(device=values.device, dtype=values.dtype)
    if weights.shape != values.shape:
        raise ValueError(
            f"{name} weights shape {tuple(weights.shape)} does not match "
            f"values shape {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError(f"{name} weights must be finite")
    if bool(weights.lt(0).any().item()):
        raise ValueError(f"{name} weights must be non-negative")
    denominator = weights.sum()
    if not bool(denominator.gt(0).item()):
        raise ValueError(f"{name} weights must have a positive sum")
    return (values * weights).sum() / denominator


def conditional_emotion_sft_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    sample_weights: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Return ``-mean(log P(E | S, A))`` with per-example token averaging."""
    scores = sequence_mean_log_likelihood(
        logits,
        labels,
        ignore_index=ignore_index,
    ).mean_log_likelihood
    return _weighted_mean(-scores, sample_weights, "conditional")


def paired_appraisal_contrastive_loss(
    *,
    score_ea_given_aa: Tensor,
    score_ea_given_ab: Tensor,
    score_eb_given_ab: Tensor,
    score_eb_given_aa: Tensor,
    beta: float = 1.0,
    pair_weights: Tensor | None = None,
) -> PairContrastiveResult:
    """Contrast matched and crossed appraisal/emotion likelihoods.

    Each score must already be the average token log-likelihood of the named
    emotion target. The four arguments correspond to::

        score_ea_given_aa = s(E_a | S, A_a)  # matched
        score_ea_given_ab = s(E_a | S, A_b)  # crossed
        score_eb_given_ab = s(E_b | S, A_b)  # matched
        score_eb_given_aa = s(E_b | S, A_a)  # crossed
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    scores = [
        score_ea_given_aa.reshape(-1),
        score_ea_given_ab.reshape(-1),
        score_eb_given_ab.reshape(-1),
        score_eb_given_aa.reshape(-1),
    ]
    shape = scores[0].shape
    if any(score.shape != shape for score in scores[1:]):
        raise ValueError("All paired score tensors must have the same shape")
    if not shape or shape[0] == 0:
        raise ValueError("Paired score tensors cannot be empty")
    if any(not bool(torch.isfinite(score).all().item()) for score in scores):
        raise ValueError("Paired score tensors must be finite")

    margin_a = scores[0] - scores[1]
    margin_b = scores[2] - scores[3]
    loss_a = F.softplus(-beta * margin_a)
    loss_b = F.softplus(-beta * margin_b)
    per_pair_loss = loss_a + loss_b
    loss = _weighted_mean(per_pair_loss, pair_weights, "pair")
    return PairContrastiveResult(
        loss=loss,
        margin_a=margin_a,
        margin_b=margin_b,
        accuracy_a=margin_a.gt(0).to(margin_a.dtype),
        accuracy_b=margin_b.gt(0).to(margin_b.dtype),
        per_pair_loss=per_pair_loss,
    )


def paired_appraisal_contrastive_loss_from_sequences(
    *,
    logits_ea_given_aa: Tensor,
    labels_ea_given_aa: Tensor,
    logits_ea_given_ab: Tensor,
    labels_ea_given_ab: Tensor,
    logits_eb_given_ab: Tensor,
    labels_eb_given_ab: Tensor,
    logits_eb_given_aa: Tensor,
    labels_eb_given_aa: Tensor,
    beta: float = 1.0,
    pair_weights: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> PairContrastiveResult:
    """Convenience wrapper that derives all four pair scores from logits."""
    score_arguments: list[Tensor] = []
    for logits, labels in (
        (logits_ea_given_aa, labels_ea_given_aa),
        (logits_ea_given_ab, labels_ea_given_ab),
        (logits_eb_given_ab, labels_eb_given_ab),
        (logits_eb_given_aa, labels_eb_given_aa),
    ):
        score_arguments.append(
            sequence_mean_log_likelihood(
                logits,
                labels,
                ignore_index=ignore_index,
            ).mean_log_likelihood
        )
    return paired_appraisal_contrastive_loss(
        score_ea_given_aa=score_arguments[0],
        score_ea_given_ab=score_arguments[1],
        score_eb_given_ab=score_arguments[2],
        score_eb_given_aa=score_arguments[3],
        beta=beta,
        pair_weights=pair_weights,
    )


def replay_preservation_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    sample_weights: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Full-chain SFT replay loss used to preserve warmup behavior.

    The caller supplies replay labels for the original complete chain. Unlike
    AE conditional labels, these labels should supervise the complete assistant
    chain selected for preservation.
    """
    scores = sequence_mean_log_likelihood(
        logits,
        labels,
        ignore_index=ignore_index,
    ).mean_log_likelihood
    return _weighted_mean(-scores, sample_weights, "preservation")


def combine_ae_losses(
    conditional_loss: Tensor,
    *,
    pair_result: PairContrastiveResult | None = None,
    preservation_loss: Tensor | None = None,
    pair_weight: float = 1.0,
    preservation_weight: float = 1.0,
) -> AELossOutput:
    """Combine AE losses while preserving each raw component for logging."""
    if conditional_loss.numel() != 1:
        raise ValueError("conditional_loss must be a scalar tensor")
    if pair_weight < 0.0 or preservation_weight < 0.0:
        raise ValueError("Loss weights must be non-negative")
    zero = conditional_loss.new_zeros(())
    pair_loss = zero if pair_result is None else pair_result.loss
    preserve = zero if preservation_loss is None else preservation_loss
    if pair_loss.numel() != 1 or preserve.numel() != 1:
        raise ValueError("pair and preservation losses must be scalar tensors")
    total = (
        conditional_loss
        + pair_weight * pair_loss
        + preservation_weight * preserve
    )

    diagnostics: dict[str, Tensor] = {
        "pair_active": conditional_loss.new_tensor(float(pair_result is not None)),
        "preservation_active": conditional_loss.new_tensor(
            float(preservation_loss is not None)
        ),
    }
    if pair_result is not None:
        diagnostics.update(
            {
                "pair_margin_a": pair_result.margin_a.mean(),
                "pair_margin_b": pair_result.margin_b.mean(),
                "pair_accuracy_a": pair_result.accuracy_a.mean(),
                "pair_accuracy_b": pair_result.accuracy_b.mean(),
                "pair_accuracy_both": (
                    pair_result.margin_a.gt(0) & pair_result.margin_b.gt(0)
                )
                .to(pair_result.margin_a.dtype)
                .mean(),
            }
        )
    return AELossOutput(
        total_loss=total,
        conditional_loss=conditional_loss,
        pair_loss=pair_loss,
        preservation_loss=preserve,
        diagnostics=diagnostics,
    )


__all__ = [
    "AELossOutput",
    "IGNORE_INDEX",
    "PairContrastiveResult",
    "SequenceLogLikelihood",
    "combine_ae_losses",
    "conditional_emotion_sft_loss",
    "extract_logits",
    "paired_appraisal_contrastive_loss",
    "paired_appraisal_contrastive_loss_from_sequences",
    "replay_preservation_loss",
    "sequence_mean_log_likelihood",
]
