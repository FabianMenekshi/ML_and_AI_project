"""
GPTQ + LQER: Advanced Post-Training Quantization.

Features:
- GPTQ with Hessian-based error compensation for 2D weight matrices
- Asymmetric quantization (zero-point offset) for better range utilization
- Group-wise quantization (GROUP=32) for finer per-group scales
- Int7 quantization for embedding matrices (more precision for tied weights)
- Int6 quantization for attention/MLP matrices
- LQER rank-4 correction on the TOP_K=3 most sensitive layers

Usage:
    from gptq_lqer import gptq_lqer_quantize_state_dict

    quant_obj, quant_stats = gptq_lqer_quantize_state_dict(
        model=base_model,
        calibration_tokens=val_tokens,
        matrix_bits=6,
        embed_bits=7,
        lqer_rank=4,
        lqer_top_k=3,
        group_size=32,
        asymmetric=True,
        device=device,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


# -------------------------
# CONSTANTS
# -------------------------

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


# =========================================================
# ASYMMETRIC GROUP-WISE QUANTIZATION
# =========================================================

def asymmetric_groupwise_quantize(
    weight: Tensor,
    quant_bits: int = 6,
    group_size: int = 32,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Asymmetric group-wise quantization.

    Instead of one scale per row (symmetric, centered at zero), this computes
    per-group scale and zero_point, allowing the quantization range to shift
    to better cover the actual weight distribution.

    Args:
        weight:     (out_features, in_features) float32 weight.
        quant_bits: target bitwidth.
        group_size: number of consecutive columns per group.

    Returns:
        q:          (out_features, in_features) int8 quantized weights.
        scale:      (out_features, num_groups) fp16 per-group scales.
        zero_point: (out_features, num_groups) int8 per-group zero points.
    """
    W = weight.float()
    out_features, in_features = W.shape
    max_val = 2 ** quant_bits - 1  # asymmetric uses [0, max_val] range

    # Pad if in_features not divisible by group_size
    num_groups = math.ceil(in_features / group_size)
    padded = num_groups * group_size
    if padded > in_features:
        W = F.pad(W, (0, padded - in_features), value=0.0)

    # Reshape into groups: (out_features, num_groups, group_size)
    W_grouped = W.reshape(out_features, num_groups, group_size)

    # Per-group min and max
    w_min = W_grouped.amin(dim=2)  # (out_features, num_groups)
    w_max = W_grouped.amax(dim=2)

    # Compute scale and zero_point
    scale = (w_max - w_min) / max_val
    scale = scale.clamp_min(1e-8)
    zero_point = torch.round(-w_min / scale).clamp(0, max_val)

    # Quantize
    q_grouped = torch.clamp(
        torch.round(W_grouped / scale.unsqueeze(2) + zero_point.unsqueeze(2)),
        0, max_val,
    )

    # Map from [0, max_val] to [-128, 127] for int8 storage
    # Shift so that the midpoint maps to 0
    q_shifted = (q_grouped - 2 ** (quant_bits - 1)).to(torch.int8)

    # Reshape back, trim padding
    q_flat = q_shifted.reshape(out_features, padded)[:, :in_features].contiguous()

    return q_flat, scale.to(INT8_PER_ROW_SCALE_DTYPE), zero_point.to(torch.int8)


def asymmetric_groupwise_dequantize(
    q: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    quant_bits: int = 6,
    group_size: int = 32,
    orig_in_features: int = None,
) -> Tensor:
    """Dequantize asymmetric group-wise quantized weights."""
    out_features, in_features = q.shape
    num_groups = scale.shape[1]
    padded = num_groups * group_size

    # Unshift from int8 back to [0, max_val]
    q_unshifted = q.float() + 2 ** (quant_bits - 1)

    # Pad if needed
    if padded > in_features:
        q_unshifted = F.pad(q_unshifted, (0, padded - in_features), value=0.0)

    q_grouped = q_unshifted.reshape(out_features, num_groups, group_size)
    scale_f = scale.float().unsqueeze(2)
    zp_f = zero_point.float().unsqueeze(2)

    W_grouped = (q_grouped - zp_f) * scale_f
    W_flat = W_grouped.reshape(out_features, padded)

    if orig_in_features is not None:
        W_flat = W_flat[:, :orig_in_features]
    else:
        W_flat = W_flat[:, :in_features]

    return W_flat.contiguous()


# =========================================================
# SYMMETRIC PER-ROW QUANTIZATION (for embeddings at int7)
# =========================================================

def symmetric_quantize(
    weight: Tensor,
    quant_bits: int = 7,
    clip_percentile: float = 99.99984,
) -> tuple[Tensor, Tensor]:
    """Symmetric per-row quantization (same logic as baseline, configurable bits)."""
    W = weight.float()
    max_val = 2 ** (quant_bits - 1) - 1
    clip_q = clip_percentile / 100.0

    clip_abs = torch.quantile(W.abs(), clip_q, dim=1)
    clipped = torch.maximum(torch.minimum(W, clip_abs[:, None]), -clip_abs[:, None])
    scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
    q = torch.clamp(
        torch.round(clipped / scale[:, None]), -max_val, max_val
    ).to(torch.int8).contiguous()

    return q, scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()


# =========================================================
# GPTQ CORE
# =========================================================

def gptq_quantize_weight(
    weight: Tensor,
    hessian: Tensor,
    quant_bits: int = 6,
    group_size: int = 32,
    asymmetric: bool = True,
    clip_percentile: float = 99.99984,
    block_size: int = 128,
    percdamp: float = 0.01,
) -> dict:
    """
    GPTQ quantization with optional asymmetric group-wise quantization.

    Returns a dict with keys:
        'q':          int8 quantized weights
        'scale':      per-row or per-group scales
        'zero_point': per-group zero points (only if asymmetric)
        'asymmetric': bool flag
        'group_size': int
        'quant_bits': int
        'orig_in_features': int (original in_features before any padding)
    """
    W = weight.float().clone()
    out_features, in_features = W.shape
    max_val_sym = 2 ** (quant_bits - 1) - 1
    max_val_asym = 2 ** quant_bits - 1

    # --- Prepare Hessian inverse ---
    H = hessian.double().clone()
    n = H.shape[0]
    diag_idx = torch.arange(n, device=H.device)

    # Handle dead channels
    dead = torch.diag(H) == 0
    if dead.any():
        H[diag_idx[dead], diag_idx[dead]] = 1.0
        W[:, dead] = 0

    # Damping with escalating retries
    mean_diag = torch.diag(H).mean().item()
    H_orig = H.clone()
    L = None
    for damp_mult in (percdamp, percdamp * 10, percdamp * 100, percdamp * 1000):
        H = H_orig.clone()
        damp = damp_mult * mean_diag
        H[diag_idx, diag_idx] += damp
        try:
            L = torch.linalg.cholesky(H)
            break
        except torch._C._LinAlgError:
            continue

    if L is None:
        raise RuntimeError("Cholesky failed — Hessian is severely degenerate.")

    H_inv = torch.cholesky_inverse(L)
    H_inv_cho = torch.linalg.cholesky(H_inv, upper=True).float()

    # --- Pre-compute quantization parameters ---
    if asymmetric:
        num_groups = math.ceil(in_features / group_size)
        padded = num_groups * group_size

        # We'll compute per-group scale/zp from the original W for each group
        # But GPTQ modifies W as it goes, so we compute scale/zp from the
        # current W values at quantization time per-block.
        # For simplicity, pre-compute from original W (standard approach).
        W_padded = F.pad(W, (0, padded - in_features), value=0.0) if padded > in_features else W.clone()
        W_grouped = W_padded.reshape(out_features, num_groups, group_size)
        w_min = W_grouped.amin(dim=2)
        w_max = W_grouped.amax(dim=2)
        group_scale = ((w_max - w_min) / max_val_asym).clamp_min(1e-8)
        group_zp = torch.round(-w_min / group_scale).clamp(0, max_val_asym)
    else:
        clip_q = clip_percentile / 100.0
        clip_abs = torch.quantile(W.abs(), clip_q, dim=1)
        row_scale = (clip_abs / max_val_sym).clamp_min(1.0 / max_val_sym)

    # --- GPTQ column-by-column ---
    Q = torch.zeros_like(W)
    num_blocks = math.ceil(in_features / block_size)

    for block_idx in range(num_blocks):
        col_start = block_idx * block_size
        col_end = min(col_start + block_size, in_features)
        block_cols = col_end - col_start

        W_block = W[:, col_start:col_end].clone()
        Err_block = torch.zeros_like(W_block)
        H_inv_block_diag = torch.diag(H_inv_cho[col_start:col_end, col_start:col_end])

        for j in range(block_cols):
            col = col_start + j
            w_col = W_block[:, j]
            d = H_inv_block_diag[j]

            # Quantize this column
            if asymmetric:
                g = col // group_size  # which group this column belongs to
                s = group_scale[:, g]
                zp = group_zp[:, g]
                q_col = torch.clamp(
                    torch.round(w_col / s + zp), 0, max_val_asym
                )
                dequant_col = (q_col - zp) * s
                # Store shifted for int8
                Q[:, col] = q_col - 2 ** (quant_bits - 1)
            else:
                s = row_scale
                clipped = torch.clamp(w_col, -clip_abs, clip_abs)
                q_col = torch.clamp(torch.round(clipped / s), -max_val_sym, max_val_sym)
                dequant_col = q_col * s
                Q[:, col] = q_col

            # Error and compensation
            err = (w_col - dequant_col) / d
            Err_block[:, j] = err

            if j + 1 < block_cols:
                W_block[:, j + 1:] -= (
                    err.unsqueeze(1) *
                    H_inv_cho[col, col_start + j + 1:col_end].unsqueeze(0)
                )

        # Inter-block compensation
        if col_end < in_features:
            W[:, col_end:] -= Err_block @ H_inv_cho[col_start:col_end, col_end:]

    q_int8 = Q.to(torch.int8).contiguous()

    result = {
        "q": q_int8,
        "quant_bits": quant_bits,
        "orig_in_features": in_features,
    }

    if asymmetric:
        result["scale"] = group_scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
        result["zero_point"] = group_zp.to(torch.int8).contiguous()
        result["asymmetric"] = True
        result["group_size"] = group_size
    else:
        result["scale"] = row_scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
        result["asymmetric"] = False

    return result


# =========================================================
# LQER: LOW-RANK QUANTIZATION ERROR RECONSTRUCTION
# =========================================================

def compute_lqer_correction(
    weight_orig: Tensor,
    weight_quantized_dequant: Tensor,
    hessian: Tensor,
    rank: int = 4,
) -> tuple[Tensor, Tensor]:
    """
    Compute the LQER low-rank correction for a quantized weight matrix.

    Uses activation-aware SVD (L²QER): scales the error by activation
    magnitudes before SVD so the correction prioritizes directions that
    matter most for the actual data.

    Args:
        weight_orig:              (out, in) original fp32 weight.
        weight_quantized_dequant: (out, in) dequantized weight after GPTQ.
        hessian:                  (in, in) Hessian H = X^T X.
        rank:                     rank of the low-rank correction.

    Returns:
        A: (out, rank) fp16 left factor.
        B: (rank, in) fp16 right factor.
        Such that weight ≈ weight_quantized_dequant + A @ B
    """
    E = (weight_orig - weight_quantized_dequant).float()
    out_features, in_features = E.shape

    # L²QER: scale by activation magnitudes (diagonal of H)
    # This steers SVD to prioritize error directions that interact with
    # large activations — same insight as AWQ but applied to the correction.
    h_diag = torch.diag(hessian).float().clamp_min(1e-8)
    activation_scale = h_diag.sqrt()  # (in_features,)

    # Scale the error: each column j of E is scaled by activation_scale[j]
    E_scaled = E * activation_scale.unsqueeze(0)

    # SVD on the scaled error
    k = min(rank, min(out_features, in_features))
    U, S, Vt = torch.linalg.svd(E_scaled, full_matrices=False)

    # Keep top-k components
    U_k = U[:, :k]           # (out, k)
    S_k = S[:k]              # (k,)
    Vt_k = Vt[:k, :]         # (k, in)

    # Undo the activation scaling on V
    # E_scaled = E * diag(s) => E = E_scaled * diag(1/s)
    # So the correction is U_k @ diag(S_k) @ Vt_k @ diag(1/activation_scale)
    Vt_k_unscaled = Vt_k / activation_scale.unsqueeze(0)

    A = (U_k * S_k.unsqueeze(0)).to(torch.float16).contiguous()   # (out, k)
    B = Vt_k_unscaled.to(torch.float16).contiguous()              # (k, in)

    return A, B


def compute_layer_sensitivity(
    weight_orig: Tensor,
    weight_quantized_dequant: Tensor,
    hessian: Tensor,
) -> float:
    """
    Compute the output reconstruction error for a layer.
    This is used to rank layers for selective LQER (TOP_K).

    Returns trace(E^T H E) which measures how much the quantization
    error affects the layer's output.
    """
    E = (weight_orig - weight_quantized_dequant).float()
    H = hessian.float()
    # Output error = trace(E @ H @ E^T) = sum of (E @ H) * E
    EH = E @ H
    error = (EH * E).sum().item()
    return error


# =========================================================
# CALIBRATION (same as before)
# =========================================================

class HessianCollector:
    def __init__(self):
        self.hessians: dict[str, Tensor] = {}
        self.num_samples: dict[str, int] = {}
        self.hooks: list = []

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
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
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_hessians(self) -> dict[str, Tensor]:
        result = {}
        for name, H in self.hessians.items():
            n = self.num_samples[name]
            result[name] = (H / n).cpu() if n > 0 else H.cpu()
        return result


def collect_hessians(
    model: nn.Module,
    calibration_tokens: Tensor,
    seq_len: int = 1024,
    num_calibration_seqs: int = 128,
    device: torch.device = torch.device("cuda"),
) -> dict[str, Tensor]:
    collector = HessianCollector()
    collector.register_hooks(model)
    model.eval()

    total_tokens = calibration_tokens.numel()
    usable = min(num_calibration_seqs * seq_len, total_tokens - 1)
    num_seqs = usable // seq_len
    batch_size = min(num_seqs, 16)

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


def build_weight_to_layer_map(model: nn.Module) -> dict[str, str]:
    mapping = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            mapping[f"{name}.weight"] = name
    return mapping


# =========================================================
# PASSTHROUGH HELPERS
# =========================================================

def keep_float_tensor(
    name: str, t: Tensor,
    passthrough_orig_dtypes: dict[str, str],
    fp32_patterns: tuple[str, ...] = CONTROL_TENSOR_NAME_PATTERNS,
) -> Tensor:
    if any(pattern in name for pattern in fp32_patterns):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


# =========================================================
# MAIN PIPELINE
# =========================================================

def gptq_lqer_quantize_state_dict(
    model: nn.Module,
    calibration_tokens: Tensor,
    matrix_bits: int = 6,
    embed_bits: int = 7,
    lqer_rank: int = 4,
    lqer_top_k: int = 3,
    group_size: int = 32,
    asymmetric: bool = True,
    clip_percentile: float = 99.99984,
    block_size: int = 128,
    percdamp: float = 0.01,
    num_calibration_seqs: int = 128,
    seq_len: int = 1024,
    device: torch.device = torch.device("cuda"),
) -> tuple[dict, dict]:
    """
    Full quantization pipeline:
    1. Collect Hessians via calibration
    2. GPTQ quantize all 2D matrices (asymmetric group-wise for attn/MLP,
       symmetric per-row for embeddings)
    3. Compute layer sensitivities and apply LQER correction to top-k layers
    4. Package everything for serialization

    Args:
        model:              trained GPT model
        calibration_tokens: 1D tensor of token IDs for calibration
        matrix_bits:        bitwidth for attention/MLP weight matrices (default: 6)
        embed_bits:         bitwidth for embedding matrix (default: 7)
        lqer_rank:          rank of LQER low-rank correction (default: 4)
        lqer_top_k:         number of layers to apply LQER correction (default: 3)
        group_size:         group size for group-wise quantization (default: 32)
        asymmetric:         use asymmetric quantization for matrices (default: True)
        clip_percentile:    percentile clipping for symmetric quantization
        block_size:         GPTQ block size
        percdamp:           GPTQ damping factor
        num_calibration_seqs: number of calibration sequences
        seq_len:            sequence length
        device:             computation device

    Returns:
        (quant_obj, stats) compatible with the baseline dequantization format,
        extended with LQER correction data.
    """
    import os
    fp32_patterns = CONTROL_TENSOR_NAME_PATTERNS

    # Step 1: Collect Hessians
    print("[GPTQ+LQER] Collecting Hessians...")
    hessians = collect_hessians(
        model, calibration_tokens,
        seq_len=seq_len,
        num_calibration_seqs=num_calibration_seqs,
        device=device,
    )
    print(f"[GPTQ+LQER] Collected Hessians for {len(hessians)} layers.")

    # Step 2: Build name mapping
    weight_to_layer = build_weight_to_layer_map(model)
    state_dict = model.state_dict()

    # Step 3: GPTQ quantize everything
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    zero_points: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict] = {}
    lqer_corrections: dict[str, dict[str, Tensor]] = {}

    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors",
         "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    # Track original weights and dequantized weights for LQER sensitivity analysis
    orig_weights: dict[str, Tensor] = {}
    dequant_weights: dict[str, Tensor] = {}
    layer_hessians: dict[str, Tensor] = {}

    gptq_count = 0
    embed_count = 0
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

        # Small tensor passthrough
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes, fp32_patterns)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        layer_name = weight_to_layer.get(name)

        # Detect embedding tensor
        is_embedding = "tok_emb" in name or "embed" in name

        if t.ndim == 2 and layer_name and layer_name in hessians:
            if is_embedding:
                # Symmetric per-row quantization at int7 for embeddings
                q, s = symmetric_quantize(t, quant_bits=embed_bits, clip_percentile=clip_percentile)
                quantized[name] = q
                scales[name] = s
                qmeta[name] = {"scheme": "per_row", "axis": 0, "bits": embed_bits}
                stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
                embed_count += 1
            else:
                # GPTQ asymmetric group-wise quantization at int6 for matrices
                H = hessians[layer_name]
                result = gptq_quantize_weight(
                    t, H,
                    quant_bits=matrix_bits,
                    group_size=group_size,
                    asymmetric=asymmetric,
                    clip_percentile=clip_percentile,
                    block_size=block_size,
                    percdamp=percdamp,
                )
                quantized[name] = result["q"]
                scales[name] = result["scale"]
                meta = {
                    "scheme": "gptq_groupwise" if asymmetric else "per_row",
                    "axis": 0,
                    "bits": matrix_bits,
                    "asymmetric": asymmetric,
                }
                if asymmetric:
                    zero_points[name] = result["zero_point"]
                    meta["group_size"] = group_size
                    meta["orig_in_features"] = result["orig_in_features"]
                qmeta[name] = meta

                payload = tensor_nbytes(result["q"]) + tensor_nbytes(result["scale"])
                if asymmetric:
                    payload += tensor_nbytes(result["zero_point"])
                stats["int8_payload_bytes"] += payload

                # Save for LQER sensitivity analysis
                if asymmetric:
                    dequant = asymmetric_groupwise_dequantize(
                        result["q"], result["scale"], result["zero_point"],
                        quant_bits=matrix_bits, group_size=group_size,
                        orig_in_features=result["orig_in_features"],
                    )
                else:
                    dequant = result["q"].float() * result["scale"].float().unsqueeze(1)

                orig_weights[name] = t.float()
                dequant_weights[name] = dequant.float()
                layer_hessians[name] = H

                gptq_count += 1
        else:
            # Fallback naive symmetric quantization
            q, s = symmetric_quantize(t, quant_bits=matrix_bits, clip_percentile=clip_percentile)
            quantized[name] = q
            scales[name] = s
            qmeta[name] = {"scheme": "per_row", "axis": 0}
            stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
            naive_count += 1

        dtypes[name] = str(t.dtype).removeprefix("torch.")

    print(f"[GPTQ+LQER] GPTQ: {gptq_count} layers, Embed: {embed_count}, Naive: {naive_count}")

    # Step 4: LQER — compute sensitivities and apply to top-k
    if lqer_rank > 0 and lqer_top_k > 0 and len(orig_weights) > 0:
        print(f"[GPTQ+LQER] Computing layer sensitivities for LQER (rank={lqer_rank}, top_k={lqer_top_k})...")

        sensitivities = {}
        for name in orig_weights:
            sens = compute_layer_sensitivity(
                orig_weights[name],
                dequant_weights[name],
                layer_hessians[name],
            )
            sensitivities[name] = sens

        # Sort by sensitivity (highest first) and pick top-k
        sorted_layers = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
        top_k_layers = [name for name, _ in sorted_layers[:lqer_top_k]]

        print(f"[GPTQ+LQER] Top-{lqer_top_k} most sensitive layers:")
        for i, (name, sens) in enumerate(sorted_layers[:lqer_top_k]):
            print(f"  {i+1}. {name}: sensitivity={sens:.4f}")

        # Compute LQER corrections for top-k layers
        for name in top_k_layers:
            A, B = compute_lqer_correction(
                orig_weights[name],
                dequant_weights[name],
                layer_hessians[name],
                rank=lqer_rank,
            )
            lqer_corrections[name] = {"A": A, "B": B}
            stats["int8_payload_bytes"] += tensor_nbytes(A) + tensor_nbytes(B)
            print(f"  LQER correction for {name}: A={A.shape}, B={B.shape}, "
                  f"cost={tensor_nbytes(A) + tensor_nbytes(B)} bytes")

    print(f"[GPTQ+LQER] Done. Total payload: {stats['int8_payload_bytes']} bytes")

    # Step 5: Package
    obj: dict[str, object] = {
        "__quant_format__": "gptq_lqer_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if zero_points:
        obj["zero_points"] = zero_points
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if lqer_corrections:
        obj["lqer_corrections"] = lqer_corrections

    return obj, stats


# =========================================================
# DEQUANTIZATION (for round-trip validation)
# =========================================================

def dequantize_gptq_lqer(obj: dict[str, object]) -> dict[str, Tensor]:
    """
    Dequantize a state dict produced by gptq_lqer_quantize_state_dict.
    Handles symmetric per-row, asymmetric group-wise, and LQER corrections.
    """
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    zero_points = obj.get("zero_points", {})
    lqer = obj.get("lqer_corrections", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        meta = qmeta.get(name, {})
        s = obj["scales"][name]

        if meta.get("asymmetric", False):
            # Asymmetric group-wise dequantization
            zp = zero_points[name]
            bits = meta.get("bits", 6)
            gs = meta.get("group_size", 32)
            orig_in = meta.get("orig_in_features", q.shape[1])
            w = asymmetric_groupwise_dequantize(q, s, zp, quant_bits=bits,
                                                 group_size=gs, orig_in_features=orig_in)
        elif meta.get("scheme") == "per_row" or s.ndim > 0:
            # Symmetric per-row
            s_f = s.to(dtype=torch.float32)
            w = (q.float() * s_f.view(q.shape[0], *([1] * (q.ndim - 1))))
        else:
            # Symmetric per-tensor
            scale = float(s.item())
            w = q.float() * scale

        # Apply LQER correction if available
        if name in lqer:
            A = lqer[name]["A"].float()
            B = lqer[name]["B"].float()
            w = w + A @ B

        out[name] = w.to(dtype=dtype).contiguous()

    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t

    return out
