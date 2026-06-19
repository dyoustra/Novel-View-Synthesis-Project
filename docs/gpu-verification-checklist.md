# GPU Verification Checklist

These are the `TODO(verify)` markers in the code. Each can only be confirmed
against the actual third-party tool, so resolve them **on the Linux/GPU box after
`./setup.sh`**, in order. None are bugs — they're integration points where the
exact third-party CLI must be confirmed before trusting it.

Find them anytime with: `grep -rn "TODO(verify)" setup.sh scripts/`

---

## 1. gsplat `--test_every` flag  (HIGHEST PRIORITY — backs eval correctness)
**File:** `scripts/04_train_gsplat.py:31`
**Why it matters:** the held-out metrics (stage 05) are only valid if gsplat
actually excludes those frames from training. We pass `--test_every N`; stage 05's
`--every` must match it.

**Verify:**
```bash
python third_party/gsplat/examples/simple_trainer.py default --help | grep -i test_every
```
- If `--test_every` exists (it's in the default Config, usually default 8): no code
  change needed. Just ensure stage 04 `--test-every` and stage 05 `--every` use the
  SAME number.
- If the flag is named differently: update the `cmd` list in `04_train_gsplat.py`
  to the correct flag name, and align stage 05's `--every`.

**Done when:** training logs show a held-out/val split and the count of training
images is less than the total registered images.

---

## 2. wild-gaussians CLI  (subcommand + flags)
**File:** `scripts/04_train_wildgs.py:22`
**Current guess:** `wild-gaussians train --data colmap --output outputs/wildgs --backend colmap`

**Verify:**
```bash
cat third_party/wild-gaussians/README.md        # find the train command
wild-gaussians --help 2>/dev/null || python -m wildgaussians --help
```
Update the `cmd` list in `04_train_wildgs.py` to match the real subcommand and flag
names (data path, output dir, COLMAP backend). It may use nerfbaselines-style
invocation rather than a bare `wild-gaussians train`.

**Done when:** a short run produces a model artifact in `outputs/wildgs/`.

---

## 3. ZeroNVS inference CLI  (most likely to need real rework)
**File:** `scripts/07_run_zeronvs.py:48`
**Current guess:** `python <ZeroNVS>/scripts/run_inference.py --image ... --azimuth ... --elevation ... --output ...`
**Reality:** ZeroNVS is threestudio-based; inference is likely a `launch.py` call
with a config + checkpoint, not a simple `run_inference.py`. Expect to rewrite the
inner `cmd`.

**Verify:**
```bash
cat third_party/ZeroNVS/README.md               # find the inference/sampling command
ls third_party/ZeroNVS/                          # locate launch.py / configs / ckpts
```
Rewrite the `cmd` in `07_run_zeronvs.py` to the real inference entrypoint, mapping:
- input image  → our `ref_img`
- target pose / azimuth+elevation → our `yaw` (and a fixed elevation)
- output path  → our `dst` (keep the `ref{r:05d}_view{v}.png` naming so figures stay
  comparable with the gsplat render in stage 06)

**Done when:** 100 PNGs land in `outputs/zeronvs/` named like the stage-06 outputs.
**Note:** ZeroNVS is the scoped *stretch* baseline — if its env/CLI fights you, land
the 3DGS half (gsplat + wild-gaussians) first; the report works without it.

---

## 4 & 5. Pin third-party commit SHAs  (do LAST, after everything works)
**File:** `setup.sh:14` (`WILDGS_SHA`) and `setup.sh:15` (`ZERONVS_SHA`)
**Why:** they're currently `"main"` (a moving target). Once a run succeeds, pin the
exact tested commits for reproducibility.

**Resolve:**
```bash
git -C third_party/wild-gaussians rev-parse HEAD   # paste into WILDGS_SHA
git -C third_party/ZeroNVS rev-parse HEAD          # paste into ZERONVS_SHA
```
Replace the `"main"` values with these SHAs and remove the `TODO(verify)` comments.

**Done when:** `setup.sh` pins all three tools (SAM2 already pinned) to fixed SHAs.

---

### Suggested order on the GPU box
1. `./setup.sh` (clones at `main`) → 2. verify gsplat `--test_every` (#1) →
3. wild-gaussians CLI (#2) → 4. full 3DGS run (COLMAP → train → eval → render) →
5. ZeroNVS CLI (#3) → 6. pin SHAs (#4/#5) and re-commit `setup.sh`.
