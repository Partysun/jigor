#!/usr/bin/env python
"""Export the von NLI checkpoint to the ONNX graph jigor's VonBackend runs.

Von (github.com/wfzyx/von, also on HF as `wfzyx/von`) is the open-source
"system one" decision model. jigor's `VonBackend` uses the NLI-era checkpoint:
a ModernBERT-large backbone plus a 3-class sequence-classification head with
labels entailment / neutral / contradiction. Each hypothesis is judged as a
(pair-encoded) NLI forward pass:

    input_ids [B, L]        ->  logits [B, 3]   (entail=0, neutral=1, contra=2)
    attention_mask [B, L]

B and L are dynamic (jigor batches all of a question's hypotheses in one
forward and truncates pairs to 512 tokens). The temperature-scaled softmax
over the entailment column is applied by jigor at runtime.

The weights live in the `wfzyx/von` repo history: revision `999e01cfff98`
("Upload folder using huggingface_hub", September 2026) still carries the
full 174-tensor sequence-classification checkpoint and the matching
`calibration.json` (`temperature: 1.0367`). Later von revisions replaced the
head with the option-marker scorer and dropped the NLI head from
`model.safetensors`. The published artifact is jigor's default:

    HF `Zatsepin/von-onnx-fp16` (`model.onnx`, `tokenizer/tokenizer.json`,
    `calibration.json`)

Export recipe:

- torch dynamo exporter, opset 18, dynamic batch/sequence axes
- `model.half()` (fp16 weights and activations), I/O dtypes preserved
- a trailing graph `Cast` turns the fp16 `logits` back to fp32 so ORT
  returns `tensor(float)` exactly like the original export

Usage:

    uv run python scripts/von/export_onnx.py \
        --out /tmp/von/model.onnx
    uv run python scripts/von/export_onnx.py --check \
        --onnx /tmp/von/model.onnx

The first run downloads `wfzyx/von` revision `999e01cfff98` (~1.6 GB, public,
Apache-2.0) into the huggingface_hub cache; pass `--revision` to pin another
commit. `--check` verifies ORT-vs-PyTorch parity on grouped 2/4-option asks
(the same grouping jigor's `noul`/`choice` softmaxes produce).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np


def download_source(revision: str, local_dir: Path) -> Path:
    """Fetch the checkpoint files for `revision` of wfzyx/von."""
    from huggingface_hub import snapshot_download

    print(f"downloading wfzyx/von @ {revision} (may take a while) ...")
    path = snapshot_download(
        repo_id="wfzyx/von",
        revision=revision,
        local_dir=local_dir,
        allow_patterns=[
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "calibration.json",
        ],
    )
    return Path(path)


def export(source: Path, out: Path) -> None:
    import onnx
    import torch
    from onnx import TensorProto, helper
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"loading checkpoint from {source} ...")
    if out.parent is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        source, attn_implementation="eager"
    )
    model.eval()
    model.half()
    tokenizer = AutoTokenizer.from_pretrained(source)
    _ = tokenizer  # loaded so the export and cache stay consistent

    raw = out.with_suffix(".f16.onnx")
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (
                torch.zeros(1, 4, dtype=torch.long),
                torch.ones(1, 4, dtype=torch.long),
            ),
            str(raw),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=18,
        )
    print(f"export took {time.perf_counter() - t0:.0f}s")

    # graph surgery: the exporter emits fp16 "logits"; rename the producer's
    # output and append a Cast->fp32 so the graph output is tensor(float).
    m = onnx.load(raw)
    g = m.graph
    cur = g.output[0].name
    producer = next(n for n in g.node if cur in n.output)
    idx = list(producer.output).index(cur)
    producer.output[idx] = cur + "_f16"
    g.node.append(helper.make_node("Cast", [cur + "_f16"], [cur], to=TensorProto.FLOAT))
    vi = helper.make_tensor_value_info(cur, TensorProto.FLOAT, ["batch", 3])
    g.output[0].CopyFrom(vi)
    del g.value_info[:]
    onnx.checker.check_model(m)
    onnx.save(m, out)
    raw.unlink(missing_ok=True)
    Path(str(raw) + ".data").unlink(missing_ok=True)
    print(f"exported {out} ({out.stat().st_size / 1e6:.0f} MB) — "
          "inputs int64, logits float32, fp16 weights")


def check(source: Path, onnx_path: Path, temp: float, n_groups: int) -> None:
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        source, attn_implementation="eager"
    )
    model.eval()
    model.half()
    tokenizer = AutoTokenizer.from_pretrained(source)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    premises = [
        "The customer asked for a refund of order #4471.",
        "Мне нужно переключить модель на giga для распознавания.",
        "We just crossed 10,000 paying customers, thank you!",
        "The disk volume /var/log is at 98% capacity and growing.",
        "напомни купить молоко и хлеб к ужину",
        "Server us-west-2 replication lag exceeded 45 seconds during the incident window.",
        "I would like to switch my writing style to a tweet mode.",
        "Это была отличная встреча, все договорились о сроках проекта.",
        "The model switch request was received and processed successfully.",
        "Не забудь отправить отчёт команде до конца дня.",
    ]
    hypotheses = [
        "The user is complaining",
        "пользователь отдаёт команду приложению",
        "The user is dictating ordinary text",
        "This incident requires operational intervention",
        "пользователь хочет переключить модель на giga",
        "The system is operating normally",
        "This is a product claim",
        "The user wants to switch to tweet mode",
        "The feature works as designed",
        "There is a critical issue with the database",
    ]

    rng = random.Random(1337)
    p_hyps = [rng.choice(premises) for _ in range(n_groups)]
    h_groups = [rng.choices(hypotheses, k=rng.choice([2, 4])) for _ in range(n_groups)]
    cases = [(p, h) for p, hs in zip(p_hyps, h_groups) for h in hs]

    enc = tokenizer(
        [c[0] for c in cases],
        [c[1] for c in cases],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        torch_logits = (
            model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
            )
            .logits.float()
            .numpy()
        )
    (ort_logits,) = session.run(
        None,
        {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
        },
    )

    def group_softmax(entail: np.ndarray) -> np.ndarray:
        out = np.zeros_like(entail)
        i = 0
        for n in map(len, h_groups):
            z = entail[i : i + n] / temp
            e = np.exp(z - z.max())
            out[i : i + n] = e / e.sum()
            i += n
        return out

    pt = group_softmax(torch_logits[:, 0])
    p16 = group_softmax(ort_logits[:, 0])
    dp = np.abs(pt - p16)
    d = np.abs(torch_logits - ort_logits)

    mism = 0
    near = 0
    i = 0
    for n in map(len, h_groups):
        a, b = np.argmax(pt[i : i + n]), np.argmax(p16[i : i + n])
        if a != b:
            top2 = np.sort(pt[i : i + n])[-2:]
            if top2[1] - top2[0] < 5e-3:
                near += 1
            else:
                mism += 1
        i += n

    print(f"raw logits max|Δ|={d.max():.2e} mean={d.mean():.2e}")
    print(f"grouped-softmax prob max|Δ|={dp.max():.2e} mean={dp.mean():.2e}")
    print(f"argmax mismatches={mism + near}/{n_groups} (near-ties={near})")
    # 1e-2 on post-temperature probabilities: fp16-vs-fp16 rounding floor;
    # the fp32 graph differs from torch at ~3e-6 (see the model card).
    if dp.max() < 1e-2 and mism == 0:
        print("PARITY OK")
    else:
        print("PARITY CHECK FAILED (tolerance 1e-2, no decisive argmax flips)")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--revision",
        default="999e01cfff98",
        help="wfzyx/von revision carrying the NLI sequence-classification head",
    )
    parser.add_argument("--source-dir", type=Path, help="already-downloaded checkpoint dir")
    parser.add_argument("--out", type=Path, default=Path("model.onnx"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--onnx", type=Path, help="onnx path to check (default: --out)")
    parser.add_argument("--temp", type=float, default=1.0367)
    parser.add_argument("--n-groups", type=int, default=60)
    args = parser.parse_args()

    source = args.source_dir or download_source(
        args.revision, Path.home() / ".cache" / "jigor" / "von-src"
    )
    if args.check:
        onnx_path = args.onnx or args.out
        check(source, onnx_path, args.temp, args.n_groups)
    else:
        export(source, args.out)


if __name__ == "__main__":
    main()