# Merging Upstream (Karpathy) Changes

## TL;DR — The 5-step process

```bash
# 1. Fetch upstream
git fetch upstream

# 2. See what changed (files + summary)
git diff HEAD..upstream/master --stat
git log HEAD..upstream/master --oneline

# 3. Pre-apply upstream diffs to OUR modified files only
git diff HEAD..upstream/master -- nanochat/gpt.py      # read this diff, apply to our gpt.py
git diff HEAD..upstream/master -- scripts/base_train.py # read this diff, apply to our base_train.py

# 4. Commit our pre-applied changes
git add nanochat/gpt.py scripts/base_train.py
git commit -m "Pre-apply upstream changes to our modified files"

# 5. Merge — take upstream for everything else, ours for our files
git merge upstream/master --no-commit
git checkout --ours nanochat/gpt.py scripts/base_train.py   # only for files with conflicts
git add -u
git merge --continue
git push origin master
```

That's it. ~5 minutes, minimal reasoning.

---

## Why this works

Our fork adds features on top of Karpathy's codebase. The vast majority of files in the repo we have **never touched** — git's automatic merge handles those perfectly (README, speedrun.sh, engine.py, fp8.py, flash_attention.py, etc.).

Only 2 files need manual attention:
- `nanochat/gpt.py` — we added `norm_mode`, `freeze_norm`, `optimizer_type`, ortho_reg support
- `scripts/base_train.py` — we added `--optimizer`, `--orth-reg-*`, `--norm-mode`, `--sv-stats-every` args

For everything else: **let git merge do its job.**

---

## Our additions vs Karpathy's additions

Our research additions live in distinct, non-overlapping areas:

| Our addition | Where in gpt.py | Conflicts with upstream? |
|---|---|---|
| `norm_mode` / `freeze_norm` in GPTConfig | Config dataclass fields | No — upstream adds new fields too |
| Conditional LayerNorm in Block | `Block.__init__` and `Block.forward` | Possible — if upstream changes Block |
| `optimizer_type='adamw'` in setup_optimizer | Bottom of `setup_optimizer` | Possible — if upstream restructures groups |
| `schedule_wd` flag for AdamW matrix params | Loop inside setup_optimizer | Possible |
| `--orth-reg-*` args + ortho_reg calls | base_train.py training loop | No — separate block after optimizer step |
| `--sv-stats-every` + sv_stats calls | base_train.py training loop | No — separate eval block |

In practice, upstream changes are **additive** (new params, new optimizer groups, new forward logic) and our additions are also **additive**. They don't overwrite the same lines. The only conflict is that git's 3-way merge gets confused when both sides added things to the same region.

**Solution: pre-apply, then use `--ours`.** Since we pre-applied upstream's changes to our files before merging, our version IS the correct merged result. `git checkout --ours` is correct.

---

## Reading the diff efficiently

When pre-applying upstream changes, only read `git diff HEAD..upstream/master -- <file>`.

**Mechanical changes** (just apply, no thinking needed):
- New hyperparameter values (e.g., `1.15 → 1.2`, `0.5 → 0.4`, `// 3 → // 4`)
- New `__init__` parameters added (e.g., `smear_gate`, `smear_lambda`)
- New init logic (non-uniform lambdas)
- New optimizer param groups added to `param_groups` list
- New forward pass logic added (smear, backout)
- New scheduler logic (momentum warmdown)

**Changes to skip / override with ours** (upstream removes our additions):
- Removal of `norm_mode` / `freeze_norm` — we keep these
- Removal of `optimizer_type` parameter — we keep this
- Removal of LayerNorm conditional in Block — we keep this
- Removal of norm param handling in setup_optimizer — we keep this

The rule: **if upstream removes something we added, keep ours. If upstream adds something new, add it to ours too.**

---

## Files we own vs files we take from upstream

### Always take from upstream (we've never modified these)
```bash
git checkout upstream/master -- nanochat/engine.py    # only if we haven't modified it
git checkout upstream/master -- runs/speedrun.sh      # only if we haven't modified it
# etc.
```
But honestly: just let `git merge` handle them. Only manually take files when there's a specific reason.

### Files we own (always use --ours after pre-applying)
- `nanochat/gpt.py`
- `scripts/base_train.py`

### Files we own entirely (no upstream equivalent)
- `nanochat/ortho_reg.py`
- `nanochat/sv_stats.py`
- `scripts/plot_sv_stats.py`
- `runs/h200_*.sh` (our runner scripts)

---

## What NOT to do

- ❌ Don't read the entire modified files — only read `git diff HEAD..upstream/master -- <file>`
- ❌ Don't use `git stash` + `git merge` (causes double-application of changes)
- ❌ Don't manually reconstruct files from diffs — apply the diff surgically with Edit tool
- ❌ Don't reason about whether Karpathy's changes are significant — just integrate them

---

## Token-efficient workflow for Claude

1. `git fetch upstream && git diff HEAD..upstream/master --stat` → which files changed?
2. `git diff HEAD..upstream/master -- nanochat/gpt.py scripts/base_train.py` → what changed in our files?
3. Apply each `+` hunk to our file (skip any `-` hunks that remove our additions)
4. Syntax check: `PYENV_VERSION=3.11.7 python3 -c "import py_compile; py_compile.compile('nanochat/gpt.py', doraise=True)"`
5. Commit + merge + push

Total reasoning budget: read 2 diffs, apply changes, done.
