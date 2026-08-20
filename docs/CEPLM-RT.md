# CEPLM-RT — Merged Training Method

**CEPLM-RT** combines two methods into one objective mixture.

## 1. CEPLM (existing)

Contract-Echo Path Language Modeling:

- **Path conditioning** — sequences prefixed with `PATH:<repo-relative-path>`
- **Contract echo** — Q↔A pairs about Entry harness laws (permissions, state machine, verification, …)

## 2. RTM (new — Rollout Trace Modeling)

Synthetic multi-step agent **execution traces** with explicit tags:

```text
[RUN] id=run_1234 task=fix sandbox resume
[STATE] RUN_CREATED
[STATE] STEP_STARTED
[STATE] MODEL_CALLED
[MODEL] plan: ...
[STATE] TOOL_REQUESTED tool=shell
[STATE] TOOL_RUNNING
[TOOL] name=shell idempotent=false
[STATE] TOOL_COMPLETED status=ok
[STATE] VERIFICATION
[VERIFY] pass=true
[STATE] COMPLETE
```

This does not exist as a standard LM objective. It teaches the model to continue a **durable run**, not only complete a sentence.

## Mixture (default)

| Share | Source |
|------:|--------|
| 40% | Path-conditioned repo code (CEPLM) |
| 30% | Contract Q↔A (CEPLM) |
| 30% | Rollout traces (RTM) |

## Recommended model: ~21M params

| | |
|--|--|
| Params | **20,943,936** (~20.94M) |
| Layers | 10 |
| Heads | 8 |
| Emb | 416 |
| Context | 256 |
| Steps | **5000** |

Better compute allocation than 50M × few hundred steps on CPU.

## Train (Termux / desktop)

```bash
bash scripts/build_corpus.sh
STEPS=5000 ACCUM=8 bash scripts/train_20m_rt.sh
```

Script: `src/ceplm_rt_20m_train.py`

## Observed (CPU sandbox run)

```
step     1  loss ~4.73
step  1000  avg ~2.16
step  2000  avg ~1.84
step  3000  avg ~1.57
step  5000  final_loss ~1.64
```

Samples begin to form permission / TOOL_STATE structure; more steps or a BPE tokenizer will improve fluency further.
