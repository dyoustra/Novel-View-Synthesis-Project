# GPU Verification Checklist

These are the `TODO(verify)` markers in the code. Each can only be confirmed
against the actual third-party tool, so resolve them **on the Linux/GPU box after
`./setup.sh`**, in order. None are bugs — they're integration points where the
exact third-party CLI must be confirmed before trusting it.

Find them anytime with: `grep -rn "TODO(verify)" setup.sh scripts/`

**Status:** ✅ 1 done · 🟡 2 CLI done (verify ingestion on first run) · ⬜ 3 ZeroNVS · ⬜ 4/5 pins

---

## 1. gsplat `--test-every` flag  ✅ DONE  (backs eval correctness)
**Status:** Resolved — verified on gsplat 1.5.3 (`--test-every`, default 8); wrapper
updated to the hyphen form; stage 05 `--every` default 8 matches the held-out split.
**File:** `scripts/04_train_gsplat.py`
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

## 2. wild-gaussians CLI  🟡 CLI DONE — verify data ingestion on first run
**File:** `scripts/04_train_wildgs.py`
**Resolved:** wild-gaussians needs CUDA 11.8 / Py3.11, so it does NOT share the
`nvs` env. It ships as a NerfBaselines method; the wrapper now runs
`nerfbaselines train --method wild-gaussians --data <colmap> --output <out>
--backend conda` from the dedicated `nb` env (README "Install" step 5).
`--backend conda` makes NerfBaselines build the isolated CUDA-11.8 env itself.

**Verify on the box (from the `nb` env):**
```bash
conda activate nb
nerfbaselines --version                                   # confirm >= 1.2.0
nerfbaselines train --help | grep -iE "method|data|output|backend"
```
The one unproven piece is whether nerfbaselines ingests our raw `colmap/` dir at
`--data`; if it complains about layout, capture the error. The first `--backend
conda` run also builds the method env (slow, one-time).

**Done when:** a short run produces a model artifact under `outputs/wildgs/`.

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

## 4 & 5. Pin remaining versions  (do LAST, after everything works)
**Where:** the `nb` env (nerfbaselines version) and `setup.sh` (`ZERONVS_SHA`).
**Why:** Method B's version *is* the nerfbaselines version (it manages wild-gaussians
internally), and ZeroNVS is still cloned at `"main"` (a moving target).

**Resolve:**
```bash
pip show nerfbaselines | grep Version             # pin this in README step 5 (>=X.Y)
git -C third_party/ZeroNVS rev-parse HEAD         # paste into ZERONVS_SHA
```
Pin the tested nerfbaselines version in the README `nb`-env step, replace
`ZERONVS_SHA="main"` with the SHA, and remove its `TODO(verify)`.

**Done when:** every tool is pinned — SAM2 SHA, gsplat `==`, nerfbaselines `>=`, ZeroNVS SHA.

---

### Suggested order on the GPU box
Walk the pipeline once and fix each **Check** as the stage that needs it comes up.
Below, "Check N" = a numbered item *in this doc*; "stage NN" = a `scripts/NN_*` step.

`./setup.sh` → **Check 1** (gsplat flag) → **Check 2** (wild-gaussians CLI) →
run stages 03–06 (COLMAP → train → eval → render) → **Check 3** (ZeroNVS) + stage 07
→ **Checks 4 & 5** (pin SHAs, re-commit `setup.sh`).
