"""
GPTQ: Post-Training Quantization with Hessian-based error compensation.

This module implements GPTQ quantization for the baseline GPT model.
It replaces the naive round-to-nearest quantization with an optimal
sequential rounding that compensates for each rounding error using
the inverse Hessian of the layer-wise reconstruction objective.

Usage:
    from gptq import gptq_quantize_state_dict

    # After training, instead of quantize_state_dict_int8:
    quant_obj, quant_stats = gptq_quantize_state_dict(
        model=base_model,
        calibration_data=val_tokens,   # or a subset
        quant_bits=8,
        device=device,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import defaultdict
import math


# -------------------------
# GPTQ CORE ALGORITHM
# -------------------------

def gptq_quantize_weight(
    weight: Tensor,
    hessian: Tensor,
    quant_bits: int = 8,
    clip_percentile: float = 99.99984,
    block_size: int = 128,
    percdamp: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """
    Quantize a 2D weight matrix using GPTQ.

    Args:
        weight:         (out_features, in_features) float32 weight matrix.
        hessian:        (in_features, in_features) Hessian approximation H = X^T X.
        quant_bits:     target bitwidth (8 → ±127, 6 → ±31, 4 → ±7).
        clip_percentile: percentile for clipping before quantization.
        block_size:     number of columns to process in each block.
        percdamp:       damping factor for Hessian diagonal (numerical stability).

    Returns:
        q:     (out_features, in_features) int8 quantized weights.
        scale: (out_features,) fp16 per-row scale factors.
    """
    W = weight.float().clone()
    out_features, in_features = W.shape
    max_val = 2 ** (quant_bits - 1) - 1

    # --- Compute per-row scale (same as baseline, before GPTQ rounding) ---
    clip_q = clip_percentile / 100.0
    clip_abs = torch.quantile(W.abs(), clip_q, dim=1)
    scale = (clip_abs / max_val).clamp_min(1.0 / max_val)

    # --- Prepare Hessian inverse ---
    H = hessian.float().clone()

    # Damping: add a small value to the diagonal for numerical stability.
    # This prevents issues when the Hessian is singular or near-singular.
    damp = percdamp * torch.diag(H).mean()
    H.diagonal().add_(damp)

    # Compute the Cholesky decomposition of H, then invert.
    # Using Cholesky because H is symmetric positive definite (after damping).
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        # Fallback: if Cholesky fails, add more damping and retry
        H.diagonal().add_(damp * 10)
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)

    # Get the Cholesky decomposition of H_inv (used for efficient block updates)
    H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)

    # --- Quantize column by column, in blocks ---
    Q = torch.zeros_like(W)
    Err = torch.zeros_like(W)

    num_blocks = math.ceil(in_features / block_size)

    for block_idx in range(num_blocks):
        col_start = block_idx * block_size
        col_end = min(col_start + block_size, in_features)
        block_cols = col_end - col_start

        # Work on this block of columns
        W_block = W[:, col_start:col_end].clone()
        Err_block = torch.zeros_like(W_block)

        # Extract the diagonal and off-diagonal parts of H_inv for this block
        H_inv_block_diag = torch.diag(H_inv_cho[col_start:col_end, col_start:col_end])

        for j in range(block_cols):
            col = col_start + j
            w_col = W_block[:, j]

            # The diagonal of H_inv tells us the quantization sensitivity
            d = H_inv_block_diag[j]

            # Quantize this column: clip, scale, round
            s = scale  # per-row scale
            clipped = torch.clamp(w_col, -clip_abs, clip_abs)
            q_col = torch.clamp(torch.round(clipped / s), -max_val, max_val)

            # Store quantized values
            Q[:, col] = q_col

            # Compute the quantization error for this column
            # Error = original_weight - dequantized_weight
            err = (w_col - q_col * s) / d

            Err_block[:, j] = err

            # Compensate remaining columns in this block
            if j + 1 < block_cols:
                W_block[:, j + 1:] -= err.unsqueeze(1) * H_inv_cho[col, col_start + j + 1:col_end].unsqueeze(0)

        # Compensate remaining blocks (columns beyond this block)
        if col_end < in_features:
            W[:, col_end:] -= Err_block @ H_inv_cho[col_start:col_end, col_end:]

    q_int8 = Q.to(torch.int8).contiguous()
    scale_fp16 = scale.to(torch.float16).contiguous()

    return q_int8, scale_fp16


# -------------------------
# CALIBRATION DATA COLLECTION
# -------------------------

class HessianCollector:
    """
    Collects input activations at each linear layer via forward hooks,
    and computes the Hessian approximation H = X^T X for GPTQ.
    """

    def __init__(self):
        self.hessians: dict[str, Tensor] = {}
        self.num_samples: dict[str, int] = {}
        self.hooks: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, name: str):
        """Create a forward hook that accumulates H = X^T X for a linear layer."""
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            # x shape: (batch, seq_len, in_features) or (batch * seq_len, in_features)
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            # Accumulate H = X^T X
            n = x.shape[0]
            if name not in self.hessians:
                self.hessians[name] = torch.zeros(
                    (x.shape[1], x.shape[1]), device=x.device, dtype=torch.float32
                )
                self.num_samples[name] = 0
            self.hessians[name].add_(x.t() @ x)
            self.num_samples[name] += n
        return hook_fn

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on all linear layers (CastedLinear and nn.Linear)."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_hessians(self) -> dict[str, Tensor]:
        """Return the averaged Hessian for each layer."""
        result = {}
        for name, H in self.hessians.items():
            n = self.num_samples[name]
            if n > 0:
                result[name] = (H / n).cpu()
            else:
                result[name] = H.cpu()
        return result


def collect_hessians(
    model: nn.Module,
    calibration_tokens: Tensor,
    seq_len: int = 1024,
    num_calibration_seqs: int = 128,
    device: torch.device = torch.device("cuda"),
) -> dict[str, Tensor]:
    """
    Run calibration data through the model and collect Hessians for all linear layers.

    Args:
        model:                the trained GPT model.
        calibration_tokens:   1D tensor of token IDs.
        seq_len:              sequence length (should match training).
        num_calibration_seqs: number of sequences to use for calibration.
        device:               device to run calibration on.

    Returns:
        Dictionary mapping layer name → Hessian tensor (in_features, in_features).
    """
    collector = HessianCollector()
    collector.register_hooks(model)

    model.eval()

    # Prepare calibration sequences
    total_tokens = calibration_tokens.numel()
    usable = min(num_calibration_seqs * seq_len, total_tokens - 1)
    num_seqs = usable // seq_len
    batch_size = min(num_seqs, 16)  # process in small batches to save memory

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            batch_end = min(batch_start + batch_size, num_seqs)
            batch_seqs = batch_end - batch_start

            start_idx = batch_start * seq_len
            end_idx = batch_end * seq_len + 1
            tokens = calibration_tokens[start_idx:end_idx].to(device=device, dtype=torch.int64)

            x = tokens[:-1].reshape(batch_seqs, seq_len)
            y = tokens[1:].reshape(batch_seqs, seq_len)
            model(x, y)

    collector.remove_hooks()
    model.train()

    return collector.get_hessians()


# -------------------------
# MAP STATE DICT NAMES TO LAYER NAMES
# -------------------------

def build_weight_to_layer_map(model: nn.Module) -> dict[str, str]:
    """
    Build a mapping from state_dict parameter name (e.g. 'blocks.0.attn.c_q.weight')
    to the module name used by forward hooks (e.g. 'blocks.0.attn.c_q').
    """
    mapping = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            param_name = f"{name}.weight"
            mapping[param_name] = name
    return mapping


# -------------------------
# FULL GPTQ QUANTIZATION PIPELINE
# -------------------------

# Re-use the same constants and helpers from the baseline
CONTROL_TENSOR_NAME_PATTERNS_DEFAULT = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(
    name: str, t: Tensor,
    passthrough_orig_dtypes: dict[str, str],
    fp32_patterns: tuple[str, ...] = CONTROL_TENSOR_NAME_PATTERNS_DEFAULT,
) -> Tensor:
    if any(pattern in name for pattern in fp32_patterns):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def gptq_quantize_state_dict(
    model: nn.Module,
    calibration_tokens: Tensor,
    quant_bits: int = 8,
    clip_percentile: float = 99.99984,
    block_size: int = 128,
    percdamp: float = 0.01,
    num_calibration_seqs: int = 128,
    seq_len: int = 1024,
    device: torch.device = torch.device("cuda"),
) -> tuple[dict, dict]:
    """
    Quantize a model's state dict using GPTQ for 2D matrices
    and baseline passthrough for small/control tensors.

    This is a drop-in replacement for quantize_state_dict_int8.

    Returns:
        (quant_obj, stats) in the same format as quantize_state_dict_int8.
    """
    import os
    control_patterns = tuple(
        p for p in os.environ.get(
            "CONTROL_TENSOR_NAME_PATTERNS",
            ",".join(CONTROL_TENSOR_NAME_PATTERNS_DEFAULT),
        ).split(",") if p
    )
    fp32_patterns = tuple(
        p for p in os.environ.get(
            "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
            ",".join(control_patterns),
        ).split(",") if p
    )

    # Step 1: Collect Hessians via calibration
    print("[GPTQ] Collecting Hessians from calibration data...")
    hessians = collect_hessians(
        model, calibration_tokens,
        seq_len=seq_len,
        num_calibration_seqs=num_calibration_seqs,
        device=device,
    )
    print(f"[GPTQ] Collected Hessians for {len(hessians)} layers.")

    # Step 2: Build mapping from weight names to layer names
    weight_to_layer = build_weight_to_layer_map(model)

    # Step 3: Quantize the state dict
    state_dict = model.state_dict()

    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors",
         "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    gptq_count = 0
    naive_count = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        # Non-float passthrough
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small tensor passthrough (fp16 or fp32 for control tensors)
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes, fp32_patterns)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        # Large float tensors: use GPTQ if Hessian is available, else naive
        stats["num_float_tensors"] += 1

        layer_name = weight_to_layer.get(name)
        if t.ndim == 2 and layer_name and layer_name in hessians:
            # GPTQ quantization
            H = hessians[layer_name]
            q, s = gptq_quantize_weight(
                t, H,
                quant_bits=quant_bits,
                clip_percentile=clip_percentile,
                block_size=block_size,
                percdamp=percdamp,
            )
            gptq_count += 1
        else:
            # Fallback to naive quantization for tensors without Hessian
            q, s = _naive_quantize(t, quant_bits, clip_percentile)
            naive_count += 1

        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    print(f"[GPTQ] Quantized {gptq_count} layers with GPTQ, {naive_count} with naive rounding.")

    obj: dict[str, object] = {
        "__quant_format__": "int8_gptq_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes

    return obj, stats


def _naive_quantize(
    t: Tensor, quant_bits: int = 8, clip_percentile: float = 99.99984
) -> tuple[Tensor, Tensor]:
    """Fallback naive quantization (same as baseline)."""
    t32 = t.float()
    max_val = 2 ** (quant_bits - 1) - 1
    clip_q = clip_percentile / 100.0

    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), clip_q, dim=1)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
        q = torch.clamp(
            torch.round(clipped / scale[:, None]), -max_val, max_val
        ).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / max_val if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(
        torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -max_val, max_val
    ).to(torch.int8).contiguous()
    return q, scale
