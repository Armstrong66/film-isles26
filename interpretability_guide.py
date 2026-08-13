"""
ISLES26 — Visualization, Interpretability & Ablation Guide
==========================================================
Template for coding assistant implementation.

DESIGN PRINCIPLES
-----------------
1. Zero training overhead  — all hooks are read-only and cost < 1ms/batch
2. Eval-time heavy lifting — UMAP, GradCAM, calibration run post-training only
3. Publication quality     — all figures target 600 DPI, LNCS-safe colour palette
4. Self-contained          — each section can be run independently

RECOMMENDED FIGURES FOR PAPER (4–6 pages)
------------------------------------------
Fig 1  — Methodology pipeline             (from isles26-diagram skill)
Fig 2  — FiLM gate analysis               (Section A)
Fig 3  — Dataset sample grid              (Section E)
Table 1 — 5-fold CV metrics               (Section D)
Table 2 — Ablation study                  (Section D)

RECOMMENDED FOR SUPPLEMENTARY / POSTER
----------------------------------------
Fig S1 — Bottleneck embedding UMAP        (Section B)
Fig S2 — GradCAM lesion attribution       (Section C)
Fig S3 — Calibration / PR curve           (Section F)
Fig S4 — Training curves per fold         (already in visualize.py)

INSTALLATION ADDITIONS
-----------------------
pip install umap-learn==0.5.6 captum==0.7.0 matplotlib==3.9.0 scikit-learn==1.5.0
(captum for GradCAM; umap-learn for embedding plots)

OUTPUT PATH: /home/derrick/projects/film-isles26/figures/interpretability/
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — FiLM Gate Analysis
# "Does the gate actually do different things per chronicity?"
# ═══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS
  Per-chronicity γ and β statistics at the bottleneck, proving the conditioning
  gate modulates features differently across disease phases. This is the single
  most important interpretability figure for the paper — it directly validates
  the core contribution.

IMPLEMENTATION GUIDE
  1. Hook registration (lightweight, add to train.py AND evaluate.py):

     class FiLMStatsHook:
         def __init__(self):
             self.records = []   # list of dicts per forward pass

         def __call__(self, gamma, beta, chronicity_batch):
             # gamma, beta: (B, C) tensors at bottleneck
             for i, chron in enumerate(chronicity_batch):
                 self.records.append({
                     "chronicity": chron,
                     "gamma_mean": gamma[i].mean().item(),
                     "gamma_std":  gamma[i].std().item(),
                     "beta_mean":  beta[i].mean().item(),
                     "beta_std":   beta[i].std().item(),
                     "gamma_vec":  gamma[i].detach().cpu().numpy(),  # full vector
                     "beta_vec":   beta[i].detach().cpu().numpy(),
                 })

  2. Inject into model.forward() after apply_film(), ONLY when a hook is set:

         if hasattr(self, '_film_hook') and self._film_hook is not None:
             self._film_hook(gamma, beta, meta_text)   # meta_text carries chronicity

  3. Run on val set for ONE fold after training:

         hook = FiLMStatsHook()
         model._film_hook = hook
         run_epoch(model, val_dl, ...)    # no grad, eval mode
         model._film_hook = None
         df_film = pd.DataFrame(hook.records)
         df_film.to_csv("figures/interpretability/film_stats.csv", index=False)

PLOTS TO GENERATE (one call: plot_film_gate_analysis(df_film, out_dir))
  Plot A1 — γ mean ± std per chronicity class (4 grouped bars, one per channel stat)
             x-axis: [acute, subacute, chronic, unknown]
             y-axis: mean γ value
             error bars: std across val scans in that class
             → shows the gate is biased differently per phase

  Plot A2 — β mean ± std per chronicity class (same structure as A1)
             → β shifts the baseline activation level per phase

  Plot A3 — Heatmap: mean γ vector (320-dim bottleneck) per chronicity class
             shape: (4 classes × 320 channels), seaborn heatmap
             → reveals which feature channels are most phase-sensitive
             → cluster channels: some will consistently activate for chronic,
                others for acute — this is the "what does the gate learn" story

  Plot A4 — γ × β joint scatter (per scan, coloured by chronicity)
             x: gamma_mean, y: beta_mean, colour: chronicity class
             → 2D fingerprint of conditioning per disease phase
             → ideally shows 4 separable clusters

TRACK C ADDITION (if Track C is trained):
  Same plots but replace γ/β extraction with sentence embedding distances:
     - Cosine similarity matrix between all val-set metadata strings
     - Coloured by chronicity — should show block structure if embeddings are
       capturing clinical semantics
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — Bottleneck Embedding Visualization
# "Do scans cluster by disease phase in latent space?"
# ═══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS
  UMAP of the conditioned bottleneck features — if the FiLM gate is working,
  acute/subacute/chronic scans should cluster differently AFTER conditioning
  but not necessarily BEFORE. Showing both before and after is the strongest
  ablation visualisation.

IMPLEMENTATION GUIDE
  1. Extract bottleneck embeddings during eval (mean-pool spatial dims):

         class BottleneckHook:
             def __init__(self):
                 self.embeddings = []
                 self.metadata   = []

             def __call__(self, x_before_film, x_after_film, chronicity, uid):
                 # x: (1, 320, H, W, D) → mean-pool → (320,)
                 self.embeddings.append({
                     "before": x_before_film.mean(dim=(2,3,4)).squeeze().cpu().numpy(),
                     "after":  x_after_film.mean(dim=(2,3,4)).squeeze().cpu().numpy(),
                     "chronicity": chronicity,
                     "uid": uid,
                 })

  2. In model.forward(), between bottleneck and apply_film:

         x_pre = x.clone()
         gamma, beta = self.conditioner(meta_vec, meta_text, x.shape[1])
         x = apply_film(x, gamma, beta)
         x_post = x
         if hasattr(self, '_embed_hook') and self._embed_hook:
             self._embed_hook(x_pre, x_post, meta_text, uid_batch)

  3. UMAP fitting (run on val set only, ~291 scans, fast):

         import umap
         reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
         emb_before = np.stack([r["before"] for r in hook.embeddings])
         emb_after  = np.stack([r["after"]  for r in hook.embeddings])
         labels     = [r["chronicity"] for r in hook.embeddings]

         umap_before = reducer.fit_transform(emb_before)
         umap_after  = reducer.fit_transform(emb_after)

PLOTS TO GENERATE (plot_umap_embeddings(umap_before, umap_after, labels, out_dir))
  Plot B1 — 2×1 UMAP side-by-side:
             Left: "Before conditioning" — points coloured by chronicity
             Right: "After conditioning" — same colouring
             → the "after" panel should show tighter phase clusters
             → if not, the gate isn't separating phases (tells you something)

  Plot B2 (optional, poster) — 3D UMAP (interactive plotly HTML)
             Colour: chronicity, Size: lesion volume (larger dot = larger lesion)
             → shows relationship between disease phase, lesion size, and
                where the model "puts" the scan in latent space
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — GradCAM Lesion Attribution
# "Where does the model look, and does conditioning change it?"
# ═══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS
  3D GradCAM saliency maps on the bottleneck features, overlaid on the T1w
  scan. Compare Track A (with conditioning) vs a baseline (no conditioning)
  to show the gate focuses attention differently per phase.

IMPLEMENTATION GUIDE (uses captum library)
  from captum.attr import LayerGradCam

  def compute_gradcam(model, image, meta_vec, meta_text, target_layer):
      # target_layer = model.bottleneck (the ResBlock before FiLM)
      gcam = LayerGradCam(
          lambda img: model(img, meta_vec, meta_text)[0][:, 1:2],  # class-1 logit
          target_layer
      )
      attribution = gcam.attribute(image, target=0)    # (1, C, H, W, D)
      # Reduce channels: mean over channel dim → (1, 1, H, W, D)
      saliency = attribution.mean(dim=1, keepdim=True)
      saliency = F.relu(saliency)
      # Upsample to image resolution
      saliency = F.interpolate(saliency, size=image.shape[2:], mode="trilinear")
      return saliency.squeeze().cpu().numpy()

  NOTE: GradCAM requires retain_graph=True — use sparingly, only on 3-5
        representative validation scans, not the full val set.
        Runtime: ~2-3 seconds per scan.

PLOTS TO GENERATE (plot_gradcam_overlay(img, mask, saliency, out_path))
  Plot C1 — 3×4 grid (3 chronicity classes × 4 planes):
             For each class, pick the median-Dice scan from val set.
             Show: axial, coronal, sagittal, + overlay panel
             Overlay: T1w greyscale + red lesion GT contour + saliency heatmap
             → Title each column: "Acute (N days)", "Chronic (N days)", etc.

  Plot C2 — Conditioning comparison (2 rows × 3 planes):
             Row 1: GradCAM WITHOUT conditioning (model._film_hook disabled,
                    or train a no-conditioning ablation model for one fold)
             Row 2: GradCAM WITH conditioning (Track A)
             → shows conditioning narrows/sharpens attention to lesion region

  NOTE FOR PAPER: GradCAM is compelling but risky to over-claim. Caption it as
  "illustrative attribution maps" rather than "ground truth explanations".
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — Ablation Study (Table + Bar Chart)
# "What does each component contribute?"
# ═══════════════════════════════════════════════════════════════════════════════
"""
ABLATION DESIGN (train fold 0 only for each variant — fast)

Variant                        | Track | FiLM gate | Boundary loss | Small-lesion wt
-------------------------------|-------|-----------|---------------|----------------
No conditioning (baseline)     |  —    |    ✗      |      ✗        |       ✗
+ Boundary focal loss          |  —    |    ✗      |      ✓        |       ✗
+ Small-lesion upweighting     |  —    |    ✗      |      ✓        |       ✓
+ FiLM gate (Track A, ours)    |  A    |    ✓      |      ✓        |       ✓
+ LLM gate (Track C, ours)     |  C    |    ✓      |      ✓        |       ✓

IMPLEMENTATION GUIDE
  The config system handles this cleanly:

  # Baseline: no conditioning, no fancy loss
  cfg = OmegaConf.load("configs/config.yaml")
  cfg.conditioning.track = "none"   # add a NullConditioner that returns gamma=1, beta=0
  cfg.loss.boundary_weight = 0.0
  cfg.loss.small_lesion_weight = 1.0
  # Run fold 0 only

  # Add a NullConditioner to conditioning.py:
  class NullConditioner(BaseConditioner):
      def __init__(self, cfg): super().__init__(embed_dim=1, hidden_dim=1)
      def _encode(self, meta_vec, meta_text): return torch.zeros(meta_vec.shape[0], 1)
      def forward(self, meta_vec, meta_text, feature_dim):
          B = meta_vec.shape[0]
          device = meta_vec.device
          return torch.ones(B, feature_dim, device=device), \
                 torch.zeros(B, feature_dim, device=device)

  Update build_conditioner() to handle track="none".

ABLATION TABLE FORMAT (LaTeX)
  \\begin{table}[t]
  \\centering
  \\caption{Ablation study (fold 0 validation, Track A model size: small).}
  \\begin{tabular}{lccccc}
  \\toprule
  Variant & Dice & PR-AUC & F1 & VolDiff & LesionΔ \\\\
  \\midrule
  Baseline (no conditioning)       & 0.xxx & 0.xxx & 0.xxx & x.x & x.x \\\\
  + Boundary focal loss            & 0.xxx & 0.xxx & 0.xxx & x.x & x.x \\\\
  + Small-lesion weighting         & 0.xxx & 0.xxx & 0.xxx & x.x & x.x \\\\
  + FiLM gate (Track A, proposed)  & \\textbf{0.xxx} & \\textbf{0.xxx} & ... \\\\
  + LLM gate  (Track C, proposed)  & 0.xxx & 0.xxx & 0.xxx & x.x & x.x \\\\
  \\bottomrule
  \\end{tabular}
  \\label{tab:ablation}
  \\end{table}

ABLATION BAR CHART (supplement to table)
  Plot D1 — Grouped bar chart: 5 variants × 5 metrics
             Use PALETTE from visualize.py, hatching for Track C bars
             Include error bars only if you run on multiple folds
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E — Dataset Sample Figure
# "What does the data look like?"
# ═══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS
  Representative T1w scans with lesion overlay per chronicity class, showing
  the visual diversity the model must handle. This goes in the Methods/Data
  section and takes roughly 1/4 page.

IMPLEMENTATION GUIDE
  Already partially in eda_atlas.ipynb (show_scan_with_mask). Promote to a
  standalone function in visualize.py:

  def plot_dataset_sample_grid(
      meta_csv_path: Path,
      proc_img_dir: Path,
      proc_mask_dir: Path,
      out_path: Path,
      n_per_class: int = 3,     # 3 examples per chronicity = 12 scans total
  ) -> None:

  Layout: 4 rows (chronicity classes) × n_per_class columns
          Each cell: axial slice at peak lesion depth, T1w + red lesion overlay
          Row labels: "Acute (≤7d)", "Subacute (8–90d)", "Chronic (>90d)", "Unknown"
          Column headers: "Small (<1mL)", "Medium (1–10mL)", "Large (>10mL)"
          → pick one scan per (chronicity × size) cell for maximum diversity

  Sample selection strategy:
      For each (chronicity, size_bin) cell:
          subset = df_les[(df_les.chronicity == c) & (df_les.size_cat == s)]
          if len(subset) == 0: use placeholder "N/A" panel
          else: pick the scan closest to the median lesion volume in that bin

  Per-panel annotation (small text below each image):
      f"Dice: {row.dice:.2f} | {row.lesion_vol_ml:.1f} mL"

  Export at 600 DPI, width = 170mm (full LNCS column)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F — Calibration and PR-AUC Analysis
# "Is the model's confidence well-calibrated?"
# ═══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS
  Precision-Recall curves per chronicity class, plus a reliability diagram
  (calibration plot). PR-AUC is an official ISLES26 metric so this directly
  connects to the leaderboard.

IMPLEMENTATION GUIDE
  Collect (gt_flat, prob_flat) pairs during evaluate.py — they are already
  computed inside evaluate_scan(). Save the soft maps per scan:

  # In evaluate_scan(), after computing soft_map:
  np.save(out_dir / f"{uid}_prob.npy", soft_map)   # only for val set

  Then in post-eval analysis:

  from sklearn.metrics import precision_recall_curve, auc, calibration_curve

  def plot_pr_curves_by_chronicity(eval_csv, prob_dir, out_path):
      df = pd.read_csv(eval_csv)
      fig, axes = plt.subplots(1, 2, figsize=(12, 5))

      # Plot 1: PR curves per chronicity
      for chron, color in zip(CHRON_CLASSES, CHRON_COLORS):
          subset = df[df.chronicity == chron]
          all_gt = []; all_prob = []
          for _, row in subset.iterrows():
              prob = np.load(prob_dir / f"{row.uid}_prob.npy")
              gt   = nib.load(mask_dir / f"{row.uid}_mask.nii.gz").get_fdata() > 0.5
              all_gt.extend(gt.ravel().tolist())
              all_prob.extend(prob.ravel().tolist())
          p, r, _ = precision_recall_curve(all_gt, all_prob)
          auc_val  = auc(r, p)
          axes[0].plot(r, p, label=f"{chron} (AUC={auc_val:.3f})", color=color)

      axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
      axes[0].set_title("PR curves by chronicity")
      axes[0].legend()

      # Plot 2: Calibration (reliability diagram)
      # Pool all val scans together
      fraction_pos, mean_pred = calibration_curve(all_gt_pooled, all_prob_pooled,
                                                   n_bins=10, strategy="uniform")
      axes[1].plot([0,1],[0,1], "k--", label="Perfect calibration")
      axes[1].plot(mean_pred, fraction_pos, "o-", label="Track A")
      axes[1].set_xlabel("Mean predicted probability")
      axes[1].set_ylabel("Fraction of positives")
      axes[1].set_title("Calibration plot")
      axes[1].legend()

  NOTE: Saving prob maps is optional (storage: ~1MB per scan × 291 = ~300MB).
        Skip if disk is tight; PR-AUC is already computed in evaluate.py.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION G — Training Stability Monitor (lightweight, runs during training)
# ═══════════════════════════════════════════════════════════════════════════════
"""
WHAT IT SHOWS
  Per-epoch FiLM gate statistics logged during training to detect instability.
  If γ collapses to 1 and β to 0, the gate is not learning (dead gate problem).
  If γ explodes, the gate is destabilising training.

IMPLEMENTATION GUIDE
  Add to run_epoch() in train.py — zero overhead, just stats collection:

  # After apply_film in model.forward() (with the hook already in place):
  # Log to history dict in train_fold():

  gamma_stats = {
      "gamma_mean": gamma.mean().item(),
      "gamma_std":  gamma.std().item(),
      "beta_mean":  beta.mean().item(),
      "beta_std":   beta.std().item(),
  }
  # Add to the `row` dict in the training loop — already saved to history.json

  ALERT CONDITIONS (log WARNING if any occur):
  - gamma_mean < 0.5 or > 2.0  → gate collapsing or exploding
  - gamma_std  < 0.01           → gate producing uniform scale (not learning)
  - beta_std   < 0.01           → gate producing uniform shift (not learning)

  Plot G1 (add to plot_training_curves in visualize.py):
      Third panel: γ_mean and β_mean over epochs, with alert thresholds marked
      → if curves stay near γ=1, β=0 throughout, the gate needs higher LR or
         different initialisation (raise this with the coding assistant)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION — generate all interpretability figures in one call
# ═══════════════════════════════════════════════════════════════════════════════
"""
Add to visualize.py:

def generate_interpretability_report(
    cfg:          DictConfig,
    fold:         int,
    model:        ISLES26Model,
    val_dl:       DataLoader,
    eval_csv:     Path,
    out_dir:      Path,
    run_gradcam:  bool = True,    # set False if time-constrained
    run_umap:     bool = True,
    n_gradcam:    int  = 3,       # scans per chronicity for GradCAM
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    print("Section A: FiLM gate analysis ...")
    df_film = collect_film_stats(model, val_dl)
    plot_film_gate_analysis(df_film, out_dir)

    if run_umap:
        print("Section B: Bottleneck embeddings ...")
        embeddings = collect_bottleneck_embeddings(model, val_dl)
        plot_umap_embeddings(embeddings, out_dir)

    if run_gradcam:
        print("Section C: GradCAM attribution ...")
        eval_df = pd.read_csv(eval_csv)
        plot_gradcam_grid(model, eval_df, cfg, out_dir, n_per_class=n_gradcam)

    print("Section E: Dataset sample grid ...")
    plot_dataset_sample_grid(
        proc_img_dir  = Path(cfg.data.processed_dir) / "images",
        proc_mask_dir = Path(cfg.data.processed_dir) / "masks",
        eval_csv      = eval_csv,
        out_path      = out_dir / "dataset_samples.png",
    )

    print("Section F: PR curves + calibration ...")
    plot_pr_curves_by_chronicity(eval_csv, out_dir / "probs", out_dir)

    print(f"All interpretability figures saved to {out_dir}")

CALL FROM EVALUATE.PY (add after evaluate_fold()):
    if args.interpret:
        generate_interpretability_report(
            cfg, fold, model, val_dl,
            eval_csv  = ckpt_dir / "eval_per_scan.csv",
            out_dir   = ckpt_dir / "interpretability",
            run_gradcam = True,
            run_umap    = True,
        )

CLI FLAG:
    parser.add_argument("--interpret", action="store_true",
                        help="Generate interpretability figures after evaluation")
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SPEED ESTIMATES (RTX workstation, fold 0 val set = 291 scans)
# ═══════════════════════════════════════════════════════════════════════════════
"""
Section A (FiLM stats)        ~2 min   — runs during normal eval pass, free
Section B (UMAP embeddings)   ~5 min   — UMAP fit on 291 × 320-dim vectors
Section C (GradCAM, 12 scans) ~2 min   — 3 scans/class × 4 classes × ~10s each
Section E (dataset samples)   ~3 min   — loading and rendering 12 NIfTIs
Section F (PR curves)         ~4 min   — requires loading 291 prob maps if saved

Total: ~16 min after evaluate.py finishes.
Set --run_gradcam=False to skip GradCAM if under time pressure.
GradCAM requires retain_graph=True — do NOT run during training, only post-hoc.
"""
