#!/usr/bin/env python3
"""
CEPLM-50M — expanded public corpus, ~50M params.
Memory-conscious CPU / Termux training (batch=1, grad accumulation).
"""
import os, math, time, random, json, argparse, gc
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F

class Config:
    vocab_size = 128
    block_size = 256
    n_layer = 10
    n_head = 8
    n_embd = 640
    dropout = 0.05
    bias = False

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=100, temperature=0.7, top_k=40):
        for _ in range(max_new):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

def encode(s):
    return [min(ord(c), 127) for c in s]

def decode(ids):
    return "".join(chr(i) if 0 < i < 128 else (" " if i == 0 else "?") for i in ids)

CONTRACTS = [
    ("List the three Entry permission modes.", "Ask | Auto Accept | Full Access"),
    ("Describe the durable run state machine.",
     "RUN_CREATED -> STEP_STARTED -> MODEL_CALLED -> TOOL_REQUESTED -> TOOL_RUNNING -> TOOL_COMPLETED -> VERIFICATION -> CONTINUE|COMPLETE|RETRY|FAILED"),
    ("Where must permissions be enforced?",
     "Below the model: Model -> Agent -> Tool Registry -> Permission Engine -> Tool Executor -> Sandbox"),
    ("Tool call contract metadata?",
     "tool_call_id, run_id, step_id, tool_name, arguments_hash, started_at, finished_at, status, result, error, attempt, idempotency_key"),
    ("How should subagents be modeled?",
     "As child runs with bounded context, token budget, cost budget, timeout, tool permissions, sandbox scope, cancellation."),
    ("Context priority order?",
     "P0 current task, P1 tool chain, P2 recent results, P3 important files, P4 memory, P5 older conversation, P6 stale tool output"),
    ("Model completion vs verification?",
     "Model completion != Task verification. Verification is a first-class harness primitive."),
    ("Gateway fallback states?",
     "REQUEST_NOT_STARTED, REQUEST_STARTED_NO_OUTPUT, STREAMING, TOOL_CALL_GENERATED, TOOL_EXECUTED, COMPLETED"),
    ("Sandbox lifecycle principle?",
     "Sandboxes are disposable. Durable state belongs to the agent run, not a specific VM."),
    ("Entry harness final loop?",
     "plan -> execute -> observe -> verify -> recover -> continue -> finish"),
    ("What is open-agents?",
     "Open-source reference app for building and running background coding agents on Vercel with durable workflows and sandboxes."),
]

def make_contract_echo():
    q, a = random.choice(CONTRACTS)
    mode = random.choice(["qa", "aq", "tag"])
    if mode == "qa":
        return f"PATH:docs/contracts\nQ: {q}\nA: {a}\n"
    if mode == "aq":
        return f"PATH:docs/contracts\nCONTRACT: {a}\nMEANS: {q}\n"
    return f"PATH:harness/contract\n# {q}\n{a}\n"

def build_paths_index(corpus):
    idx, marker, pos = [], "===== FILE: ", 0
    while True:
        i = corpus.find(marker, pos)
        if i < 0:
            break
        j = corpus.find(" =====", i)
        if j < 0:
            break
        path = corpus[i + len(marker):j].strip()
        next_i = corpus.find(marker, j)
        end = next_i if next_i >= 0 else len(corpus)
        idx.append((path, i, end))
        pos = j + 1
    return idx

def make_path_chunk(corpus, paths_index, block):
    if not paths_index:
        s = random.randint(0, max(0, len(corpus) - block))
        return corpus[s:s + block]
    path, lo, hi = random.choice(paths_index)
    if hi - lo < 32:
        s = random.randint(0, max(0, len(corpus) - block))
        chunk = corpus[s:s + block]
    else:
        w = random.randint(lo, max(lo, hi - block // 2))
        chunk = corpus[w:w + block]
    return (f"PATH:{path}\n" + chunk)[: block * 2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_train.txt")
    ap.add_argument("--out", default="checkpoints_50m")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    cfg = Config()

    print("=" * 60, flush=True)
    print("CEPLM-50M — ~49.4M params, CPU/Termux", flush=True)
    print("=" * 60, flush=True)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sample = Path("data/corpus_sample.txt")
        if sample.exists():
            print(f"WARN: {args.corpus} missing; using sample. Run scripts/build_corpus.sh", flush=True)
            corpus_path = sample
        else:
            raise SystemExit(f"Missing corpus: {args.corpus}. Run: bash scripts/build_corpus.sh")

    corpus = corpus_path.read_text(encoding="utf-8", errors="replace")
    paths_index = build_paths_index(corpus)
    print(f"Corpus: {len(corpus):,} chars | paths: {len(paths_index)}", flush=True)

    model = TinyGPT(cfg).to(device)
    nparams = count_params(model)
    print(f"Params: {nparams:,} ({nparams/1e6:.2f}M)", flush=True)
    print(f"L={cfg.n_layer} H={cfg.n_head} D={cfg.n_embd} T={cfg.block_size}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def one_example():
        if random.random() < 0.30:
            text = make_contract_echo()
        else:
            text = make_path_chunk(corpus, paths_index, cfg.block_size)
        ids = encode(text)
        need = cfg.block_size + 1
        if len(ids) < need:
            ids = ids + [0] * (need - len(ids))
        if len(ids) > need:
            start = random.randint(0, len(ids) - need)
            ids = ids[start:start + need]
        x = torch.tensor([ids[:cfg.block_size]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:need]], dtype=torch.long, device=device)
        return x, y

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    losses = []
    model.train()
    opt.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for step in range(1, args.steps + 1):
        x, y = one_example()
        _, loss = model(x, y)
        (loss / args.accum).backward()
        accum_loss += loss.item()

        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(accum_loss / args.accum)
            accum_loss = 0.0

        if step % 50 == 0 or step == 1:
            recent = losses[-10:] if losses else [loss.item()]
            avg = sum(recent) / len(recent)
            print(f"step {step:5d}/{args.steps}  loss {loss.item():.4f}  avg {avg:.4f}  {time.time()-t0:.0f}s", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            for prompt in [
                "PATH:docs/contracts\nQ: List the three Entry permission modes.\nA:",
                "PATH:packages/agent\nexport async function",
            ]:
                idx = torch.tensor([encode(prompt)], dtype=torch.long)
                out = model.generate(idx, max_new=100, temperature=0.6, top_k=30)
                sample = decode(out[0].tolist())[:280]
                print("--- sample ---", flush=True)
                print(sample.replace("\n", "\\n"), flush=True)
                print("--------------", flush=True)
            model.train()

        if step % args.save_every == 0 or step == args.steps:
            path = os.path.join(args.out, f"ceplm50m_step{step}.pt")
            torch.save({
                "model": model.state_dict(),
                "cfg": {k: getattr(cfg, k) for k in ["vocab_size","block_size","n_layer","n_head","n_embd","dropout","bias"]},
                "step": step,
                "loss": loss.item(),
                "nparams": nparams,
                "method": "CEPLM-50M",
            }, path)
            print(f"saved {path}", flush=True)
            gc.collect()

    summary = {
        "method": "CEPLM-50M",
        "nparams": nparams,
        "steps": args.steps,
        "final_loss": losses[-1] if losses else None,
        "corpus_chars": len(corpus),
        "device": "cpu",
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE", json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
