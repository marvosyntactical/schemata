# schemata.html — implementation plan

Heavily adapted successor to orbit.html.  Same aesthetic (warm paper tones, dark stage,
monospace sidebar), but the interaction model shifts from "explore one attractor" to
"understand a model that learns attractors sequentially."

---

## 0. What orbit.html already gives us (don't re-implement)

- Full AL-RNN forward + BPTT in vanilla JS (`ALRNN` class, ~200 lines)
- RK4 integrator, Lyapunov spectrum (variational + QR), fixed-point Newton, DMD,
  correlation dimension D₂, power spectrum, phase-portrait 3D via Three.js
- Modified Gram-Schmidt QR (already there — reuse for QR-init)
- mathjs for matrix eigenvalues (reuse for φ linear-core eigenvalues)
- Beautiful collapsible sidebar, info-drawer, responsive layout

Key gap: the JS `ALRNN` uses the full W (M×M), not the low-rank factored form
W = W_B @ W_A (r×M each).  That is the only structural change needed to the model.

---

## 1. Name and purpose

**schemata.html** — a continual-learning playground for AL-RNNs.

Tagline: *train, forget, remember*

Audience: the researcher building PIAGETS/SLAO.  Every internal quantity visible,
every hyperparameter tweakable, all in a single self-contained HTML file.

---

## 2. Layout

```
┌──────────────────────────────────────────────────────┬───────────────────────┐
│  TASK STRIP                                          │  SIDEBAR (420 px)     │
│  [●Lor28]→[Tor382]→[VdP1.5]→[Lor35]→…   [▶ Next]   │                       │
├─────────────────────────────────┬────────────────────┤  I   STREAM & CL      │
│                                 │                    │  II  MODEL            │
│   MAIN 3D PHASE PORTRAIT        │  φ-EMBEDDING PANEL │  III WEIGHT ANATOMY   │
│   (Three.js, full-height left)  │  (canvas, upper Δ) │  IV  SNAPSHOT CACHE   │
│                                 │                    │  V   ANALYSIS         │
│                                 ├────────────────────│                       │
│                                 │  MSE / LOSS CURVES │                       │
│                                 │  (canvas, lower Δ) │                       │
└─────────────────────────────────┴────────────────────┴───────────────────────┘
HUD (bottom-left): task t, λ_eff, mode (ASSIM/ACCOM), mse_train, n_regions
```

The stage is split **2:1** left:right.  Left = the existing 3D canvas (full height).
Right = two stacked secondary canvases (φ-embedding top, MSE/loss bottom), togglable
by clicking a tab bar above them.

The **task strip** is a narrow bar across the full top, ~44px, showing the CL stream as
a horizontal ribbon.  On mobile this collapses to a single "Task N of T" chip.

---

## 3. Sections

### I — Stream & CL (replaces "System")

**Task strip (persistent, outside sidebar):**
- Each task is a pill: class-coloured dot + name ("Lor28"), greyed-out = not yet reached
- Active task glows
- Click past task → overlay its oracle phase portrait on the main canvas
- Hover → tooltip: class, system params, episodes trained, final mse

**Presets** (dropdown):
- *Standard 10-task* (the experiments stream: Lor28 → Tor382 → VdP1.5 → Lor35 → ...)
- *Lorenz ramp* (Lor28 → Lor35 → Lor40 → Lor45, monotone difficulty)
- *Class alternating* (Lor / Tor / VdP / Lor / Tor / VdP ...)
- *Pathological* (easy VdP15 at position 2 — exercises the known probe failure)
- *Custom* → open a task-builder modal: pick system, params, class label, n_epochs

**CL method controls:**
- Use CL: toggle (if off → vanilla sequential fine-tuning)
- Use B_merge: toggle (SLAO merge on/off)
- Use QR-init: toggle
- Fisher type: Prediction / Signature / None
- λ: slider 0–20, current value shown in HUD
- Protect A: checkbox; Protect h: checkbox
- Adaptive λ: toggle → shows λ_assim / λ_accom sliders + ratio threshold

**Buttons:**
- `Train this task` — train on current task with current settings
- `Next task ›` — advance to next task in stream (trains if not already done)
- `Auto-run stream` / `Pause` — runs all remaining tasks one by one with animation
- `Reset model` — reinitialise fresh model (with seed), keep stream position
- `Reset all` — fresh model AND reset stream to task 0

---

### II — Model (replaces the Model section)

**Architecture params:** M, P, rank r (new: the factored W_B@W_A), latent dim
**Training params:** seq len, batch, τ (teacher forcing stride), lr, iters per task

Display-only badges:
- `n_params: 634`  `rank: 6`  `n_regions_visited: 4`

**EWC status readout** (appears after task 0):
```
EWC active   W_B ✓   A ✓   h ✗
λ_eff = 10.0  (ASSIMILATION)
F_B.max = 1.00   F_A.max = 0.83
```

**Training log** (mini mono-out box, scrollable, last 20 lines):
```
[t=3 Lor35]  ep 100/300  recon=0.09  ewc=0.01
  [probe] mse_new=1.20  ratio=1.19  → ASSIMILATION  λ=10.0
[t=3 Lor35]  ep 200/300  recon=0.11  ewc=0.01
```

**Loss curve canvas** (existing, extended): two traces — recon loss (clay) and EWC loss
(blue-bright) — so the relative magnitudes are always visible.

---

### III — Weight Anatomy (new section)

Three sub-tabs toggled by buttons in the section header:

**W tab** — heatmap of the effective matrix W = W_B @ W_A (M×M), value-coloured
(diverging blue–white–clay).  Overlay the Fisher importance as opacity of each cell:
high-importance cells are fully opaque, low-importance cells are translucent.  If
B_merge is active, a second "merged W" heatmap is shown side-by-side with delta
highlighting cells that changed.

**A / h tab** — two side-by-side bar charts: A (M-dim diagonal, shows memory
timescales), h (M-dim bias).  Each bar coloured by magnitude, EWC protection shown as a
small lock icon above bars with F_A > median.

**Φ-factors tab** — visualise the factored W_B (r×M) and W_A (M×r) as two smaller
heatmaps, left and right of a "×" symbol.  Useful for understanding which input
directions (W_A rows) combine with which output channels (W_B cols).

Implementation: all three are 2D canvas draws, no Three.js needed.  Redrawn each time
a task completes or when "Refresh" is clicked.

---

### IV — φ-Embedding (new section, also drives the right secondary canvas)

The **φ-bar panel** (in sidebar and mirrored in right canvas, upper half):

24 vertical bars in 5 colour-coded groups:

```
eigenvals (10)   act_mean (6)   act_std (6)   topo (1)   lyap (1)
  ████████████    ▓▓▓▓▓▓         ▓▓▓▓▓▓         █          ▒
```

Two overlaid bar sets: **current model** (clay) and **nearest oracle** (blue).  When
multiple oracle snapshots are cached, toggle which ones to show as ghost bars.

Fisher importance shown as a glow/halo behind each bar — literally the sensitivity
||∂φ_k/∂W_B||: which φ-dimensions are W_B-sensitive and therefore "counted" the most
in the EWC penalty.

Clicking a bar opens an info drawer explaining that φ-component (what it measures,
expected range for each attractor type, how it's computed).

The **φ-PCA scatter** (in right canvas, lower half, or a separate overlay):

- Axes: PC1 vs PC2 of φ across all oracle models (computed once at startup)
- Oracle models as labelled dots, coloured by class (Lorenz=blue, Torus=orange, VdP=green)
- Current model's φ as a moving dot, leaving a tail (path through φ-space as training progresses)
- Class centroid ellipses (σ contours from per-class oracle φ distribution)
- When probe fires: a flash animation from "before QR-init" to "after training" φ

This panel is the single most informative continuous-learning visualisation in the tool.

---

### V — Linear Subregions (new section + phase portrait overlay)

The AL-RNN is piecewise-linear: it has 2^P possible activation regions, defined by
which of the P nonlinear units have z_i > 0.  In practice only a few are visited on any
given attractor (n_regions = 4 for Lorenz, 14 for Torus).

**In the main phase portrait:**
- Colour each trajectory point by its activation pattern hash (π(z) ∈ {0,1}^P → hue)
- Different regions = different colours → you literally see the piecewise-linear segments
- When the model free-runs, watch the coloring evolve as the trajectory visits regions

**In the sidebar:**
- Donut chart: fraction of trajectory time spent in each visited region
- Table of visited regions: binary code, dwell time %, dominant eigenvalues in that region
- "Show boundaries" toggle: project the switching hypersurfaces z_i = 0 as faint planes
  onto the 3D view.  For M=16, P=6 this means up to 6 planes (one per nonlinear unit),
  which is already meaningful.

This is computationally cheap (evaluate sign of z_i at each step), visually distinctive,
and directly illustrates the model structure researchers explain in talks.

---

### VI — Snapshot Cache (new section)

Up to 6 model snapshots, each shown as a card:

```
┌─────────────────────────────────┐
│ [●] t=3 after Lor35  clay ████  │  ← colour swatch from region colouring
│ mse=0.53  BWT=−   φ: [sparkbar] │  ← mini φ-bar (10+6+6+1+1 = 24 cells)
│ [Load]  [Compare]  [Delete]     │
└─────────────────────────────────┘
```

- **Load**: restore model weights into the active model slot
- **Compare**: overlay this model's phase portrait on the main canvas (up to 3 overlaid)
- **Auto-snapshot**: toggle to automatically snapshot after each task completes

**φ-overlap view**: when ≥ 2 snapshots are selected for comparison, add their φ dots to
the φ-PCA scatter with connecting arrows showing the trajectory each model took through
φ-space.

**MSE matrix view**: a small heatmap (n_tasks × n_cached_models) showing reconstruction
MSE.  This is the BWT_mse discussed in post_meeting.md.  Rows = tasks, columns = model
snapshots.  Diagonal = freshly-trained MSE; off-diagonal = forgetting.  Coloured green
(low) to red (high).

---

### VII — Analysis (from orbit.html, adapted)

Keep: Lyapunov spectrum, fixed points, DMD/Koopman, D₂, power spectrum, sensitive
dependence.

Add:
- **Source**: now can run on "current model", "oracle (task i)", or "cached snapshot k"
- **Free-run vs data MSE**: report the actual BWT_mse metric (run each cached oracle
  system through the model and compute MSE)
- **n_regions visited**: report count + list activation patterns, already computed
  during the free-run for the subregion overlay

---

## 4. Training backend — Python/PyTorch (primary)

The JS-only finite-difference Fisher is both slower and noisier than the exact autograd
version.  The whole point of the signature Fisher is the chain rule through `diff_phi`'s
computational graph (eigenvalue decomposition, Lyapunov power iteration, soft gates).
Finite differences can fail near eigenvalue degeneracies and near the log/norm operations
in the Lyapunov code.  A Python backend is the right call.

**Split of responsibilities:**

| Layer | What lives here |
|-------|-----------------|
| `server.py` (Python/PyTorch) | All computation: ALRNN, training, EWC, QR-init, B_merge, exact signature Fisher, φ, free-run, MSE evaluation, oracle pre-training, snapshot I/O |
| `schemata.html` (JS + Three.js) | Rendering only: 3D phase portrait, all canvas widgets, WebSocket client, UI controls |

The HTML file is self-contained and works without any Python (it shows a "not connected"
banner and disables training controls), but all model operations require the server.

**Running:**
```bash
cd schemata
pip install fastapi uvicorn websockets  # torch already installed
python server.py            # starts ws://127.0.0.1:8765/ws
# open schemata.html in browser
```

**WebSocket protocol (JSON):**

Client → Server:
```
{"op":"ping"}
{"op":"init", "M":16, "P":6, "rank":6, "lam":5.0}
{"op":"set_stream", "preset":"standard_10"}
{"op":"set_cl", "use_cl":true, "use_merge":true, "use_qr":true,
                "fisher":"sig", "adaptive":false,
                "lam":5.0, "lam_assim":10.0, "lam_accom":2.0, "assim_ratio":10.0}
{"op":"train", "task_idx":0, "epochs":300}
{"op":"next"}                   ← advance + train next untrained task
{"op":"freerun", "steps":3000}
{"op":"get_weights"}
{"op":"get_phi"}
{"op":"get_fisher"}             ← exact autograd, not finite diff
{"op":"get_oracle_phis"}        ← trains oracle per task, returns φ + PCA vecs
{"op":"get_mse_matrix"}
{"op":"snapshot", "slot":0, "label":"t=3 Lor35"}
{"op":"load_snapshot", "slot":0}
{"op":"get_data", "task_idx":0}
{"op":"reset"}
{"op":"reset_all"}
```

Server → Client (streamed):
```
{"type":"pong"}
{"type":"ready", "M":16, "P":6, "rank":6}
{"type":"stream_info", "tasks":[...], "preset":"standard_10"}
{"type":"progress", "ep":50, "total":300, "recon":0.091, "ewc":0.012}
{"type":"probe", "mode":"assimilation", "lam":10.0, "mse_new":1.20, "ratio":1.19}
{"type":"task_done", "t":3, "name":"Lor_r35", "phi":[...24...],
                     "n_regions":4, "xyz":[...flat...], "regions":[...],
                     "mse":0.53, "WB":[...], "WA":[...], "A":[...], "h":[...],
                     "FB":[...], "FA":[...], "Fh":[...]}
{"type":"freerun", "xyz":[...flat...], "regions":[...ints...], "n_regions":4}
{"type":"weights", "WB":[...], "WA":[...], "A":[...], "h":[...],
                   "FB":[...], "FA":[...], "Fh":[...], "M":16, "P":6, "rank":6}
{"type":"phi", "phi":[...24...],
               "groups":{"eigenvals":[...10...], "act_mean":[...6...],
                         "act_std":[...6...], "topo":[float], "lyap":[float]}}
{"type":"fisher_done", "FB":[...], "FA":[...], "Fh":[...]}
{"type":"oracle_phis", "phis":[{"idx":0,"phi":[...],"name":"Lor_r28","cls":0,"color":"..."},...],
                        "pca_mean":[...24...], "pca_vecs":[[...24...],[...24...]]}
{"type":"mse_matrix", "current":[{"task_idx":0,"name":"Lor_r28","mse":0.53},...],
                       "snapshots":{"0":[mse,...],...},
                       "tasks":["Lor_r28",...]}
{"type":"snapshot_saved", "slot":0, "label":"t=3 Lor35", "phi":[...], "mse":0.53}
{"type":"data", "task_idx":0, "name":"Lor_r28", "xyz":[...flat...], "color":"#3e6e8e"}
{"type":"status", "msg":"...", "level":"info|warn|error"}
{"type":"reset_ok", "task_idx":0}
{"type":"cl_updated", "use_cl":true}
```

**Why no WebGPU / JS training fallback:**
For M=16, P=6 the bottleneck is not FLOPS but rather the quality of the Fisher.
Finite-difference Fisher for r=12 requires 192 φ-evaluations, each involving a
T_track=200-step autonomous rollout + eigenvalue computation.  The result is noisier
than autograd by an amount that depends on ε, can miss narrow gradient spikes, and
breaks down near degenerate eigenvalues.  The existing Python code is battle-tested.
Rewriting backprop for the full diff_phi graph in JS is not worth the correctness cost.

---

## 5. Implementation phases

### Phase 1 — Core structure (est. 1–2 days)

1. Copy orbit.html to schemata.html; rename title, colours, branding
2. **Add factored ALRNN**: replace `W: Float64Array(M*M)` with `W_B, W_A: Float64Array(r*M)`,
   update `step()`, `train()` backprop to chain rule through the product
3. **Task strip UI**: horizontal pill ribbon at top, driven by a `STREAM` array
4. **Section I** (Stream & CL): preset selector, CL toggles, λ slider, Next/Reset buttons
5. **EWC in training loop**: after task 0, add `ewc_loss` term; accumulate pred-Fisher
6. **QR-init**: call `mgsQR` on W_A rows at start of each task t > 0
7. **B_merge**: EMA store/restore on W_B; display merged W_B in sidebar

Deliverable: end-to-end CL training loop with vanilla/baseline/pred_ewc modes.

### Phase 2 — Visualisation (est. 1–2 days)

8. **Weight Anatomy** panel: canvas heatmaps for W, A, h, Fisher overlay
9. **φ-bar chart**: 24-bar canvas widget, current model + oracle comparison
10. **Linear subregion colouring**: hash activation pattern → hue, colour trajectory points
    in Three.js (`BufferAttribute` vertex colours, update per free-run step)
11. **Snapshot cache** with load/compare cards
12. **MSE matrix** heatmap (tiny grid canvas in Analysis section)
13. **HUD extensions**: λ_eff, mode (ASSIM/ACCOM), n_regions, mse_train

### Phase 3 — φ-PCA and advanced viz (est. 1 day)

14. **φ-PCA scatter** (2D canvas, power-iteration PCA as in orbit.html but on oracle φ-vectors)
15. **Signature Fisher** via finite differences (replace pred-Fisher when use_sig toggled)
16. **φ-component info drawer** entries (24 tooltip explanations)
17. **Adaptive λ probe** in JS: MSE-ratio criterion, ref_max tracking
18. **"Pathological" preset** and task builder modal

### Phase 4 — Polish + optional Python backend (est. 1 day)

19. **WebGPU** matrix multiply (progressive enhancement)
20. **Python backend** server.py with WebSocket, model import/export
21. **Animated task transitions**: probe fires → flash animation in φ-PCA; B_merge → animate
    weight heatmap morphing from current to merged
22. **Export**: snapshot as JSON (weights + φ + metadata); orbit.html-style CSV download
    for trajectory data

---

## 6. Key implementation notes

### Factored AL-RNN backprop

Forward: `z' = A⊙z + (W_B @ W_A) @ φ(z) + h`

The chain rule for the product:
```
dL/dW_B = (dL/dz') @ (W_A @ φ(z))^T     (r×M)
dL/dW_A = W_B^T @ (dL/dz') @ φ(z)^T     (M×r)
```
This is a straightforward extension of the current BPTT in orbit.html.

### φ computation (JS port of diff_phi)

```javascript
function diffPhi(model, T_warmup, T_track, beta) {
  // (a) warmup — no grad needed, just rollout
  let z = new Float64Array(M);
  for (let t = 0; t < T_warmup; t++) model.step(z, z, phiBuf);
  
  // (b) track rollout — record soft activation rates
  const softActs = [];  // T_track × P
  for (let t = 0; t < T_track; t++) {
    model.step(z, z, phiBuf);
    const row = new Float64Array(P);
    for (let i = 0; i < P; i++) row[i] = 1 / (1 + Math.exp(-beta * z[M-P+i]));
    softActs.push(row);
  }
  
  // (c) linear-core eigenvalues: A_diag[P:] + W[P:,P:]
  //     → use mathjs eigs() on the (M-P)×(M-P) submatrix
  
  // (d) activation rates (mean over time), std over time
  
  // (e) topological entropy: spectral radius of soft transition matrix T = S_{t}^T S_{t+1} / T_track
  
  // (f) max Lyapunov: same power-iteration loop as orbit.html lyapunovODE but for the
  //     piecewise-linear map (use soft gates from softActs[t])
  
  return phi;  // Float64Array(24)
}
```

The Lyapunov code is already in orbit.html for ODE systems.  The Jacobian for the
discrete-time AL-RNN map at step t is:
`J_t = diag(A) + diag(gate_t) @ W`
where `gate_t[i] = 1 if i < M-P else σ(β·z_i)` (soft).
This is structurally identical to the `matvec_rows(J, v, n)` pattern already in the file.

### Subregion colouring

```javascript
function regionHash(z, M, P) {
  let code = 0;
  for (let i = 0; i < P; i++) code |= (z[M-P+i] > 0 ? 1 : 0) << i;
  return code;  // integer 0..2^P-1
}

// map to hue: spread 2^P codes uniformly around colour wheel
function regionColor(code, P) {
  const hue = (code / (1 << P)) * 360;
  return hslToRgb(hue, 0.75, 0.65);
}
```

In Three.js: use a `BufferGeometry` with per-vertex colours for the trajectory line.
Update colours during free-run by writing to `geometry.attributes.color`.

Switching hyperplanes: for the i-th nonlinear unit (i ∈ {M-P, …, M-1}), the switching
surface is {z : z_i = 0}.  Project into the current PCA view as a line (2D slice of the
hyperplane).  Compute it as the null intersection of `z_i = 0` with the 3D PCA subspace:
this is a 2D plane in PCA coords, shown as a faint semi-transparent polygon.

### MSE matrix canvas

```
rows = tasks 0..T-1,  cols = cached snapshots (+ current model)
cell (t, k) = MSE of snapshot k on task t's data
colour = viridis-like gradient, clamped 0..max_mse
```

Small canvas, ~200×120 px.  Annotate diagonal with a corner triangle (= "trained here").
BWT_mse = mean of (off-diagonal same-task row) − diagonal.

### φ-PCA scatter animation

Maintain a `phiHistory: Float64Array[]` array — one entry per task completed.
After each task, compute φ of the current model and append.
In the scatter, draw a polyline through phiHistory dots, animate the last dot moving
toward its final position during training (re-compute φ every N epochs and update).
Oracle dots (fixed) use larger circles with class colour.

---

## 7. Style notes

Extend the orbit.html colour palette:

```css
--class-lorenz: #3e6e8e;    /* same as --blue */
--class-torus:  #b98e3e;    /* same as --ochre */
--class-vdp:    #6f7355;    /* same as --moss */
--assimilation: #6f7355;    /* moss = "settled, familiar" */
--accomodation: #b65d3a;    /* clay = "disruption, new" */
--region-base:  0.75 sat, 0.65 lig;  /* bright enough against --night */
```

Task strip pills: `border-radius: 12px`, background `var(--paper-3)` for done tasks,
`var(--clay)` border for active, `#2a2520` for future.

The φ-bar chart should feel like a **fingerprint**: thin bars, tightly packed,
very readable at a glance.  Groups separated by a 4px gap, labelled with tiny mono
text above.  Current model in clay, oracle/cached in blue-bright (semi-transparent).

The MSE matrix and weight heatmaps should use the same diverging colour scale:
`var(--moss)` for low, `--paper` for mid, `var(--clay-deep)` for high.  This keeps
everything in the same warm palette rather than importing d3/viridis.

---

## 8. Non-goals / out of scope for v1

- Multi-model side-by-side training (would require multiple ALRNN instances in Workers)
- Mobile-first layout (research tool, desktop only)
- Exporting videos / animations
- The Python backend is optional; the HTML must work standalone

---

## 9. Open questions before starting

1. **Rank r in the UI**: default to r=6 (faster Fisher) or r=12 (matches experiments)?
   Suggest exposing r as a slider 1–16, default 6 with a note "12 used in experiments."

2. **Oracle pre-computation**: the φ-PCA requires oracle φ-vectors.  Since we have no
   Python backend in standalone mode, either (a) compute oracles in the browser at startup
   (takes ~30s for 10 oracles at 300 epochs each), or (b) hard-code the oracle φ-vectors
   from the experiments as JSON constants.  Recommend option (b) for the standard presets,
   option (a) for custom task streams.

3. **B_merge EMA coefficient**: should λ_merge be a control or fixed (e.g. 1/t)?  Expose
   as a slider for exploration; default to 1/t (equal-weighted mean).

4. **Finite-difference Fisher resolution**: for signature Fisher, how many φ-evaluations
   do we parallelize?  Web Workers can't be nested, but we can batch the finite-difference
   columns in the main worker.  98 cols × 30ms/φ = ~3s total.  Acceptable for "after task"
   computation; set expectation in the UI with a progress bar.

5. **'New methods' ablation from post_meeting.md**: should the UI expose the `use_slao`
   flag (QR-init toggle) and allow "pure EWC without SLAO" directly?  Yes — this is
   trivially a checkbox already planned in §I controls.
