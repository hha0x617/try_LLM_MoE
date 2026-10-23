"""
Measure the "answer accuracy" of checkpoints with a fact-completion test. (macOS version of ../eval_facts.py)

  python macosx/eval_facts.py                                   # Evaluate out/ckpt_best.pt
  python macosx/eval_facts.py --ckpt out/ckpt_best.pt --ckpt out/ckpt.pt   # Compare several side by side
  python macosx/eval_facts.py --verbose --json out/eval_facts_baseline.json   # Save the results as JSON

Differences from ../eval_facts.py:
  - uses the Apple GPU (MPS) when available, otherwise the CPU
  - imports macosx/moe_model.py (real-valued RoPE)

For each line of eval_facts.jsonl, e.g. {"prompt": "日本の首都は", "answers": ["東京", ...]},
compute from the logits the probability that the answer immediately follows the prompt
(no sampling is involved, so the results are deterministic).

  Accuracy    : fraction where greedy decoding (temperature 0) outputs every token of the answer exactly
  Top-10      : fraction where the first token of the answer ranks in the top 10 of the 32,000-token vocabulary
  Median rank : median rank of the first token of the answer. Whitespace-only tokens such as the "▁" in " 1945" are skipped
  Answer prob.: geometric mean of the probability assigned to the full answer (= exp(mean log prob))
When there are several candidate answers, the one with the highest probability is used.
"""
import argparse
import json
import math
import os
import statistics

import sentencepiece as spm
import torch

from moe_model import load_checkpoint


def load_facts(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_ids(sp, prompt, answer):
    """Tokenize prompt + answer together and return the answer's token ids."""
    p = [sp.bos_id()] + sp.encode(prompt)
    full = [sp.bos_id()] + sp.encode(prompt + answer)
    if full[:len(p)] != p:
        # If the boundary tokens got merged, treat the common prefix as the prompt
        k = next(i for i, (a, b) in enumerate(zip(p, full)) if a != b)
        return full[:k], full[k:], False
    return p, full[len(p):], True


@torch.no_grad()
def score(model, sp, prompt, answer, device):
    ctx, ans, clean = split_ids(sp, prompt, answer)
    ids = torch.tensor([ctx + ans], device=device)
    logits = model(ids)["logits"][0, len(ctx) - 1:-1].float()  # positions that predict each answer token
    logp = logits.log_softmax(-1)
    target = torch.tensor(ans, device=device)
    tok_logp = logp.gather(1, target[:, None]).squeeze(1)
    # " 1945" splits into ['▁', '1', '9', ...], so skip whitespace-only tokens and rank the first content token
    j = next((i for i, t in enumerate(ans) if sp.id_to_piece(t) != "▁"), 0)
    first_rank = int((logits[j] > logits[j, ans[j]]).sum().item()) + 1
    return {
        "answer": answer,
        "logp": tok_logp.sum().item(),
        "n_tokens": len(ans),
        "greedy": bool((logits.argmax(-1) == target).all().item()),
        "rank": first_rank,
        "top": [sp.id_to_piece(i) for i in logits[j].topk(3).indices.tolist()],
        "clean_split": clean,
    }


def evaluate(model, sp, facts, device):
    rows = []
    for f in facts:
        cands = [score(model, sp, f["prompt"], a, device) for a in f["answers"]]
        best = max(cands, key=lambda c: c["logp"])
        best["greedy"] = any(c["greedy"] for c in cands)  # correct if any spelling is produced
        rows.append({**f, **best})
    return rows


def summarize(rows):
    n = len(rows)
    return {
        "n": n,
        "acc": sum(r["greedy"] for r in rows) / n,
        "top10": sum(r["rank"] <= 10 for r in rows) / n,
        "median_rank": statistics.median(r["rank"] for r in rows),
        "geo_prob": math.exp(sum(r["logp"] for r in rows) / n),
    }


def main():
    ap = argparse.ArgumentParser(description="Measure answer accuracy with a fill-in-the-blank fact test")
    ap.add_argument("--ckpt", action="append", default=None,
                    help="Specify multiple times to compare checkpoints (default: out/ckpt_best.pt)")
    ap.add_argument("--tokenizer", default=os.path.join("data", "tokenizer.model"))
    ap.add_argument("--facts", default="eval_facts.jsonl")
    ap.add_argument("--verbose", action="store_true", help="Also show per-question results")
    ap.add_argument("--json", default=None, help="Path to save the results as JSON")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    facts = load_facts(args.facts)
    ckpts = args.ckpt or [os.path.join("out", "ckpt_best.pt")]
    print(f"{args.facts}: {len(facts)} questions (ja {sum(f['lang'] == 'ja' for f in facts)} / "
          f"en {sum(f['lang'] == 'en' for f in facts)})")

    results = []
    for path in ckpts:
        model, ckpt = load_checkpoint(path, device)
        model.eval()
        rows = evaluate(model, sp, facts, device)
        summary = {lang: summarize([r for r in rows if r["lang"] == lang]) for lang in ("ja", "en")}
        summary["all"] = summarize(rows)
        results.append({"ckpt": path, "step": ckpt["step"], "summary": summary, "rows": rows})
        n_merged = sum(not r["clean_split"] for r in rows)
        if n_merged:
            print(f"  Note: in {path}, {n_merged} questions had the prompt/answer boundary merged into a single token")
        if args.verbose:
            print(f"\n=== {path} (step {ckpt['step']}) ===")
            for r in rows:
                mark = "○" if r["greedy"] else ("△" if r["rank"] <= 10 else "×")
                print(f"  {mark} [{r['lang']}] {r['prompt']} → {r['answer'].strip()}  "
                      f"rank {r['rank']:>5}  prob {math.exp(r['logp']):.4f}  top: {' '.join(r['top'])}")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"\n{'Checkpoint':<22}{'Step':>7}{'Lang':>6}{'Accuracy':>10}{'Top-10':>8}{'Median rank':>13}{'Answer prob.':>14}")
    for res in results:
        for lang in ("ja", "en", "all"):
            s = res["summary"][lang]
            print(f"{os.path.basename(res['ckpt']):<22}{res['step']:>7}{lang:>6}{s['acc']:>10.0%}{s['top10']:>8.0%}"
                  f"{s['median_rank']:>13.0f}{s['geo_prob']:>14.4f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.json}")


if __name__ == "__main__":
    main()
