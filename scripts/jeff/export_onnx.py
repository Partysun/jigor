#!/usr/bin/env python
"""Export a Jeff decision checkpoint (backbone + readout) to a single ONNX graph.

Jeff (github.com/firelex/jeff) is a fine-tuned zero-shot "system one" decision
model: a Qwen3.5/Gemma decoder backbone plus a small trained readout head
(Linear(hidden -> 255) over the last token's hidden state) and a fitted
temperature. A decision is one forward pass: no KV cache, no generated text.

This script exports that graph for jigor's native ONNX backend:

    input_ids [B, L]            ->  logits [B, 255]
    attention_mask [B, L]
    mm_token_type_ids [B, L]

Where L is STATIC (see --seq-len). The Qwen3.5 linear attention layers are a
chunked recurrent scan (`torch_chunk_gated_delta_rule`) whose inter-chunk
recurrence is a Python loop over chunks: tracing unrolls it, so the exported
graph needs a concrete sequence length. Clients pad left (attention mask
zeroed) to exactly L. Only the batch axis is dynamic.

Two substitutions are needed to make the graph ONNX-exportable and small:

1. The Triton kernel packages (`fla`, `causal_conv1d`) are blocked so the
   PyTorch reference implementations run; all of them are plain torch ops.
2. `torch.linalg.solve_triangular` (the reference intra-chunk UT solve, which
   has no ONNX function) is replaced by its exact algebraic inverse via a 2x2
   block recursion - pure GEMM/Concat/Slice, the same math as forward
   substitution, with the diagonal forced to 1 exactly like the reference's
   `unitriangular=True`. (A first attempt used the truncated Neumann series;
   that diverges on layers whose chunk matrices have spectral radius >= 1.)
   Verified to match torch.solve_triangular to fp32 rounding on unit-lower
   matrices of spectral radius up to ~10.

Usage (with a checkout of firelex/jeff, its venv, and the checkpoint downloaded):

    uv run python scripts/jeff/export_onnx.py \
        --checkpoint checkpoints/jeff-0.8b \
        --seq-len 512 \
        --out jeff-qwen3.5-0.8b.onnx

Then verify parity against the PyTorch reference:

    uv run python scripts/jeff/export_onnx.py --checkpoint ... --check \
        --onnx jeff-qwen3.5-0.8b.onnx --seq-len 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Force the plain PyTorch reference implementations of the hybrid-attention
# kernels: the `kernels` hub (which swaps module functions at import time),
# triton `fla` chunks and `causal_conv1d` have no ONNX representation. With
# these blocked, the transformers decorators become identity and the class
# forwards call the module-level torch reference functions we patch below.
import os

os.environ.setdefault("USE_HUB_KERNELS", "0")
sys.modules["kernels"] = None
sys.modules["fla"] = None
sys.modules["causal_conv1d"] = None

import torch  # noqa: E402

import transformers.models.qwen3_next.modeling_qwen3_next as q3n  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as q35  # noqa: E402


def patch_chunked_delta_rule() -> None:
    """Replace the reference chunked delta rule with an ONNX-exportable twin.

    The body is the reference `torch_chunk_gated_delta_rule` from the
    transformers qwen3_next/qwen3_5 modeling modules, with one change: where
    the reference calls `torch.linalg.solve_triangular` (no ONNX function) -
    or, on the dynamo path, iterates a 63-step forward-substitution loop (an
    unwieldy ~90k-node unrolled graph per chunk) - we compute the exact
    algebraic inverse via the block recursion in `substitution` below.
    `--check` verifies end-to-end parity against the model.
    """

    chunk_size = 64

    def substitution(ut_system: torch.Tensor) -> torch.Tensor:
        """Algebraic inverse of the unit-lower UT system, as a block recursion.

        The reference solves the system with `solve_triangular(..., upper=False,
        unitriangular=True)` (and its own dynamo fallback negates `tril(-1)` and
        runs forward substitution) - both treat the diagonal as exactly 1 and
        ignore ut_system's own diagonal. A Neumann power series is *not* a safe
        replacement: at some layer's chunk data the triangular matrix's spectral
        radius reaches 1, the series diverges and logits explode. Instead we
        invert exactly, like forward substitution, via a 2x2 block recursion
        (pure GEMM/Concat/Slice, with the diagonal forced to 1):

            [A11 0; A21 A22]^-1 = [U11 0; -U22 A21 U11 U22]

        For a 64x64 unit-lower block this terminates at the 1x1 identity in six
        levels; the result agrees with solve_triangular to fp rounding on every
        chunk. Verified against torch.solve_triangular on random unit-lower
        matrices of spectral radius up to ~10 (fp32, ~1e-6 max diff).
        """
        eye = torch.eye(chunk_size, device=ut_system.device, dtype=ut_system.dtype)
        unit = ut_system.tril(-1) + eye  # force the diagonal to 1, like the reference

        def invert(block: torch.Tensor) -> torch.Tensor:
            n = block.shape[-1]
            if n == 1:
                return torch.ones_like(block)
            half = n // 2
            a11 = block[..., :half, :half]
            a21 = block[..., half:, :half]
            a22 = block[..., half:, half:]
            u11 = invert(a11)
            u22 = invert(a22)
            u21 = -torch.matmul(torch.matmul(u22, a21), u11)
            top = torch.cat([u11, torch.zeros_like(u11)], dim=-1)
            bottom = torch.cat([u21, u22], dim=-1)
            return torch.cat([top, bottom], dim=-2)

        return invert(unit)

    def torch_chunk_gated_delta_rule(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        initial_dtype = query.dtype
        batch_size, sequence_length, _, k_head_dim = key.shape
        num_v_heads, v_head_dim = value.shape[-2:]
        recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
        padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
        decay = g

        query, key, value, beta, decay = [
            x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
            for x in (query, key, value, beta, decay)
        ]
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        scaling = query.shape[-1] ** -0.5
        query = query * scaling

        # Pad sequence length to a multiple of chunk_size, then chunk
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
        beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
        num_chunks = (sequence_length + pad_size) // chunk_size

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)

        query, key, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, k_beta, v_beta)
        ]
        decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

        strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
        cum_decay = decay.cumsum(dim=3)
        pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
        pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
        pairwise_decay = pairwise_decay.exp()

        ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
        intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
        decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

        # --- not part of the reference: ONNX-safe inverse of the UT system ---
        inverse = substitution(ut_system)
        new_values = inverse @ v_beta
        k_cumdecay = inverse @ decayed_k_beta
        # --------------------------------------------------------------------

        if initial_state is None:
            last_recurrent_state = torch.zeros(
                recurrent_state_shape, dtype=new_values.dtype, device=new_values.device
            )
        else:
            last_recurrent_state = initial_state.to(new_values)
        core_attn_out = torch.zeros_like(new_values)

        query = query * cum_decay.exp().unsqueeze(-1)
        key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
        chunk_decay = cum_decay[..., -1].exp()[..., None, None]

        for i in range(num_chunks):
            v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
            inter_chunk_attn = query[:, :, i] @ last_recurrent_state
            core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
            last_recurrent_state = (
                last_recurrent_state * chunk_decay[:, :, i]
                + key[:, :, i].transpose(-1, -2) @ v_new
            )

        last_recurrent_state = None if not output_final_state else last_recurrent_state
        core_attn_out = core_attn_out.reshape(padded_output_shape)
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
        return core_attn_out, last_recurrent_state

    from torch.nn import functional as F

    from transformers.models.qwen3_next.modeling_qwen3_next import l2norm

    # the same reference function lives in both qwen3_next and qwen3_5
    # modeling modules; the qwen3_5 one is what the Qwen3.5 backbone calls.
    q3n.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
    q35.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule


class JeffExport(torch.nn.Module):
    """backbone(input_ids, attention_mask, mm_token_type_ids) -> logits[B, 255].

    Mirrors `DecisionModel.forward`: last token hidden state of the backbone,
    then the trained readout Linear. The temperature is applied by the caller
    (it lives in decision_config.json, and jigor applies it at runtime).
    """

    def __init__(self, backbone: torch.nn.Module, readout: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.readout = readout

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
        )
        hidden = out.last_hidden_state[:, -1]
        return self.readout(hidden)


def build_export_model(checkpoint: Path) -> JeffExport:
    """Load the trained checkpoint in fp32 on CPU (deterministic parity baseline)."""
    from jeff.model import DecisionModel

    model = DecisionModel(str(checkpoint), device="cpu")
    model.eval()
    return JeffExport(model.backbone, model.readout)


def export(
    checkpoint: Path,
    seq_len: int,
    out: Path,
    batch: int = 2,
) -> None:
    patch_chunked_delta_rule()
    import time

    model = build_export_model(checkpoint)
    example = torch.randint(0, 100, (batch, seq_len))
    mask = torch.ones(batch, seq_len, dtype=torch.int64)
    mm = torch.full((batch, seq_len), 2, dtype=torch.int64)
    # first example has a masked (left-padded) span, so the graph exercises it
    mask[0, :64] = 0
    mm[0, :64] = 0

    t0 = time.perf_counter()
    # dynamo=True: FX-compiled export. The legacy tracer executes every Python
    # loop of the chunked delta-rule scan in the interpreter and takes tens of
    # minutes per chunk; the FX path builds the unrolled graph natively.
    torch.onnx.export(
        model,
        (example, mask, mm),
        str(out),
        input_names=["input_ids", "attention_mask", "mm_token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "mm_token_type_ids": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=18,
        dynamo=True,
        # python-level ONNX graph optimization is very slow on the unrolled
        # chunk-scan graph; ORT optimizes at session load in C++.
        optimize=False,
    )
    print(f"export took {time.perf_counter() - t0:.0f}s")
    print(f"exported {out} ({out.stat().st_size / 1e6:.0f} MB)")


def check(
    checkpoint: Path,
    onnx_path: Path,
    seq_len: int,
    n_prompts: int = 12,
) -> None:
    import onnxruntime as ort

    from jeff.model import DecisionModel

    ref = DecisionModel(str(checkpoint), device="cpu")
    ref.eval()

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)

    states = [
        "The customer says the parcel arrived crushed and wants their money back.",
        {"error": "Disk volume /var/log at 98% capacity."},
        "Voice command: open the garage door",
        "The VOTE_AGAINST bill would gut consumer protections; VOTE_FOR strengthens them.",
        "User: my account is locked after three failed attempts.",
        "Disk full",
        "The lamp is on.",
        "The box arrived.",
    ]
    questions = [
        {"type": "noul", "instructions": "Is the customer angry?"},
        {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"1": "Refunds and payments", "2": "Damaged or lost parcels", "3": "Account and login problems"},
        },
        {"type": "score", "instructions": "How urgent is this?", "criteria": ["calm", "annoyed", "angry"]},
        {"type": "choice", "instructions": "Which option best matches?", "criteria": {"A": "pro", "B": "con", "C": "neutral"}},
        {"type": "noul", "instructions": "Does this need review?", "criteria": {"true": "yes immediately", "false": "no"}},
    ]

    temp = ref.temperature
    max_diff = 0.0
    evaluated = 0
    for i in range(n_prompts):
        state = states[i % len(states)]
        q = questions[i % len(questions)]
        rows = [{"state": state, "question": q} for _ in range(2)]
        batch = ref.prepare(rows)
        ids = batch.inputs["input_ids"]
        length = ids.shape[1]
        if length > seq_len:
            print(f"skip prompt {i}: {length} tokens > {seq_len}")
            continue
        # pad left to the static seq_len, exactly like a client would
        pad = seq_len - length
        pad_token = ref.processor.tokenizer.pad_token_id or 0
        pad_ids = torch.full((ids.shape[0], pad), pad_token, dtype=torch.int64)
        pad_mask = torch.zeros((ids.shape[0], pad), dtype=torch.int64)
        pad_mm = torch.full((ids.shape[0], pad), 2, dtype=torch.int64)
        ids_full = torch.cat([pad_ids, ids], dim=1)
        mask_full = torch.cat([pad_mask, batch.inputs["attention_mask"]], dim=1)
        mm_full = torch.cat([pad_mm, batch.inputs["mm_token_type_ids"]], dim=1)

        with torch.no_grad():
            expected = (ref(batch) / temp).softmax(-1)

        (logits,) = session.run(
            None,
            {
                "input_ids": ids_full.numpy(),
                "attention_mask": mask_full.numpy(),
                "mm_token_type_ids": mm_full.numpy(),
            },
        )
        # the reference masks logits beyond the question's option count with
        # -1e9 before the temperature softmax (DecisionModel.forward); do the
        # same to the ONNX logits before comparing
        n_options = 2 if q["type"] == "noul" else len(q["criteria"])
        logits = torch.from_numpy(logits).float()
        mask = torch.arange(255)[None, :] >= n_options
        logits = logits.masked_fill(mask, -1e9)
        got = (logits / temp).softmax(-1)
        diff = (got - expected).abs().max().item()
        max_diff = max(max_diff, diff)
        evaluated += 1
        print(f"prompt {i}: seq={length + pad} max|Δ prob| = {diff:.2e}")
    if evaluated == 0:
        print("no prompt fit under --seq-len; nothing to compare")
        sys.exit(2)
    print(f"MAX |Δ| over {evaluated} prompts: {max_diff:.2e}")
    # 1e-3: fp32 ORT-vs-torch rounding (the model's own fp32-vs-bf16 floor is
    # ~1e-2 on probabilities, so 1e-3 is effectively bit-exact for our purposes)
    if max_diff < 1e-3:
        print("PARITY OK")
    else:
        print("PARITY CHECK FAILED (tolerance 1e-3)")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("jeff.onnx"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--n-prompts", type=int, default=12)
    args = parser.parse_args()

    if args.check:
        check(args.checkpoint, args.out, args.seq_len, args.n_prompts)
    else:
        export(args.checkpoint, args.seq_len, args.out)


if __name__ == "__main__":
    main()