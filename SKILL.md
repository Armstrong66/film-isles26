---
name: isles26-diagram
description: Generate a publication-quality, editable draw.io architecture diagram for the ISLES26 MICCAI/SWITCH+ paper. Use this skill whenever the user asks to draw, diagram, visualise, or illustrate the ISLES26 model, methodology, pipeline, architecture, or any component thereof (preprocessing, conditioning, FiLM gate, LLM conditioning, training regime, evaluation). Also triggers for "figure for the paper", "methodology figure", or "architecture figure". The skill uses the Agents365-ai drawio-skill under the hood and knows the full ISLES26 project context — no re-explanation needed.
---

# ISLES26 Architecture Diagram Skill

Generates a compact, research-grade draw.io diagram of the ISLES26 methodology
for inclusion in the MICCAI SWITCH+ 2026 paper (4–6 page Springer LNCS limit).
Outputs an editable `.drawio` XML file and a high-DPI PNG ready for LaTeX `\includegraphics`.

## Prerequisites

This skill wraps the Agents365-ai drawio-skill. Install it first if not already present:

```bash
# Clone into your skills directory
git clone https://github.com/Agents365-ai/drawio-skill.git \
  ~/.claude/skills/drawio-skill

# Install draw.io desktop CLI (Linux/RTX workstation)
# Download the .deb from https://github.com/jgraph/drawio-desktop/releases
sudo apt install xvfb   # headless export support
xvfb-run -a drawio --version   # verify ≥ 30
```

## Project context (do not ask the user to re-explain)

Read `~./CLAUDE.md` for full context.
Key facts to embed in the diagram:

**Dataset:** ATLAS v3.0, N=1453 native-space skull-stripped T1w MRI, 33 sites.
**Task:** Binary ischemic stroke lesion segmentation in native space.

**Pipeline stages (left → right, top-level flow):**
1. Raw T1w MRI (native space, variable orientation/spacing)
2. Preprocessing: LAS→RAS reorientation · per-scan foreground z-score · patch crop 128³
3. Metadata encoding: DAYS_POST_STROKE (log-norm) + CHRONICITY_DERIVED (4-class) → 5-dim vector OR natural language string
4. 3D U-Net encoder: 1→16→32→64→128→256 channels, strided conv downsampling
5. Bottleneck: 256→320 channels, ResBlock
6. **Conditioning injection** (the core contribution): FiLM gate (Track A) OR sentence-transformer gate (Track C) applied to bottleneck features
7. 3D U-Net decoder: 4-scale upsampling with skip connections + deep supervision heads
8. Loss: Dice + CE + Boundary Focal + small-lesion upweighting
9. Training: 5-fold CV, poly LR, AdamW, mixed precision
10. Post-processing: connected-component filtering (min 10 voxels)
11. Output: binary lesion mask (native space, matched geometry)
12. Evaluation: Dice, PR-AUC, Lesion-wise F1, Abs Vol Diff, Abs Lesion Count Diff

**Two conditioning tracks (show both in diagram):**
- Track A — FiLM: 5-dim metadata vector → MLP → (γ, β) scale-shift → applied to bottleneck
- Track C — LLM gate: natural language string → frozen sentence-transformer (all-MiniLM-L6-v2, 22MB) → mean-pool → projection → (γ, β)

**Key novelty to highlight visually:**
The conditioning gate modulates the decoder differently per chronicity phase
(acute/subacute/chronic/unknown), enabling a single model to adapt its
decoding strategy based on disease stage.

## Diagram variants to generate

### Variant 1 — Full methodology pipeline (recommended for Figure 1)
A horizontal left-to-right pipeline showing all stages from raw data to evaluation.
Compact enough to fit in a half-page figure (≤ 85mm height for single-column LNCS).
Use the ML/Deep Learning preset from drawio-skill.

**Layout:** 3 rows
- Row 1 (top): Data → Preprocessing → Metadata encoding
- Row 2 (middle, emphasis): 3D U-Net (encoder → bottleneck → **conditioning gate** → decoder) → Loss → Output
- Row 3 (bottom): Evaluation metrics bar

Chronicity-specific augmentation shown as a callout off the preprocessing box.

### Variant 2 — Conditioning module detail (for Figure 2 or supplementary)
A zoomed-in view of just the conditioning gate, showing:
- Left branch: Track A FiLM path
- Right branch: Track C sentence-transformer path
- Both converging to the same (γ, β) → feature modulation interface
- Bottleneck feature map shown as a 3D tensor block
- FiLM equation: x̂ = γ ⊙ x + β annotated on the arrow

### Variant 3 — Results overview (for Results section)
A 2×2 grid of metric boxes (Dice by chronicity class) — use the data preset.
Skip this if training results are not yet available.

## Generation instructions

When the user asks for a diagram, do this:

1. **Identify which variant** they want (default: Variant 1).

2. **Load the drawio-skill** from `~/.claude/skills/drawio-skill/skills/drawio-skill/SKILL.md`.

3. **Issue the generation prompt** to drawio-skill using the ML/Deep Learning preset.
   Use this exact prompt structure (fill in details from the context above):

   ```
   Create a publication-quality ML methodology diagram using the ML/Deep Learning
   preset. Horizontal left-to-right layout. Compact for half-page LNCS figure
   (target width 170mm for double-column, height ≤ 90mm).

   [Insert specific variant description from above]

   Style requirements:
   - White/light grey background (#FFFFFF / #F8F8F8)
   - Color scheme (LNCS-safe, prints well in greyscale):
     - Data/IO boxes: #E8F4FD (light blue)
     - Model architecture boxes: #EBF5EB (light green)
     - Conditioning gate boxes: #FFF3E0 (light orange) — highlight the novelty
     - Loss/training boxes: #F3E5F5 (light purple)
     - Evaluation boxes: #FFF8E1 (light yellow)
     - Track A (FiLM): #FF8C00 accent
     - Track C (LLM): #6A0DAD accent
   - Font: Helvetica or Arial, 8–9pt for labels, 7pt for annotations
   - Arrows: orthogonal routing, 1.5pt weight
   - No drop shadows (they don't print well at small sizes)
   - Dashed border on the conditioning gate box to signal "pluggable module"
   - Bold border on the U-Net block to signal the backbone
   - Annotation: "(γ, β)" on the FiLM injection arrow
   - Small "(×5 folds)" annotation near the training block
   ```

4. **Export settings** — instruct drawio-skill to export as:
   - `.drawio` file (editable, save to project root as `figures/isles26_architecture.drawio`)
   - PNG at 600 DPI (for LaTeX: `figures/isles26_architecture.png`)
   - If CLI unavailable, generate XML only and provide the diagrams.net import URL

5. **Self-check** — after export, verify:
   - All text is readable at the target print size
   - Conditioning gate is visually prominent (it is the contribution)
   - Greyscale version still distinguishable (important for black-and-white printing)
   - Diagram fits within LNCS column width constraints

6. **Iterate** up to 3 rounds based on user feedback.

## Output paths

```
C:\Users\DELL\Downloads\film-isles26\
└── figures/
    ├── isles26_architecture.drawio      # editable source
    ├── isles26_architecture.png         # 600 DPI for LaTeX
    ├── isles26_conditioning_detail.drawio
    └── isles26_conditioning_detail.png
```

## LaTeX inclusion snippet (provide to user after export)

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/isles26_architecture.png}
  \caption{Overview of the proposed chronicity-conditioned segmentation pipeline.
           The metadata conditioning gate (orange, dashed border) is the core
           contribution, modulating decoder features per disease phase via
           FiLM (Track~A) or a frozen sentence transformer (Track~C).}
  \label{fig:architecture}
\end{figure}
```

## Fallback: XML-only mode (no draw.io CLI)

If draw.io CLI is not installed on the current machine, generate the `.drawio`
XML directly and output it as a code block. The user can:
1. Copy the XML
2. Open https://app.diagrams.net
3. File → Import From → Device → paste XML
4. Export → PNG at 600 DPI

In XML-only mode, still apply all style requirements above — embed them as
cell styles in the XML.

## Size constraints summary (for LNCS compliance)

| Figure position | Max width | Max height | Recommended DPI |
|---|---|---|---|
| Full-width (double col) | 170mm | 90mm | 600 |
| Half-width (single col) | 83mm | 80mm | 600 |
| Supplementary | 170mm | 140mm | 300 |

Keep the methodology figure full-width. Keep the conditioning detail half-width
or full-width depending on complexity. Never exceed these — LNCS margins are strict.
