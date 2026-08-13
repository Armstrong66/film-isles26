# drawio-skill Integration for ISLES26

This document describes how to generate architecture diagrams for the ISLES26 project.

## Current Status

The `drawio-skill` referenced in SKILL.md is **not installed** in this environment.
The skill references paths like `~/.claude/skills/drawio-skill` which don't exist
on Windows.

## Diagram Generation Options

### Option 1: Manual diagrams.net (Recommended for now)

1. Open https://app.diagrams.net
2. Import the XML template (see below)
3. Export as PNG at 600 DPI for LaTeX

### Option 2: Install drawio-skill locally

```bash
# Clone into your skills directory (requires Git)
git clone https://github.com/Agents365-ai/drawio-skill.git \
  ~/.claude/skills/drawio-skill

# Install draw.io desktop (Windows)
# Download from: https://github.com/jgraph/drawio-desktop/releases
# Then run:
drawio --version
```

### Option 3: Use Python to generate draw.io XML

I can generate a complete `.drawio` XML file directly in this conversation.
You can then:
1. Copy the XML
2. Open https://app.diagrams.net
3. File → Import From → Device → paste XML
4. Export → PNG at 600 DPI

## Output Directory

All figures should be saved to:
```
C:\Users\DELL\Downloads\film-isles26\figures\
```

## Pipeline Verification

The following pipeline components are verified and up-to-date:

- **Preprocessing**: reorient_to_ras, clip_and_normalise, resample_to_shape
- **Augmentation**: MONAI transforms + AcuteContrastReduction + ChronicCavityPerturbation
- **Dataset**: ISLES26Dataset with metadata encoding (5-dim for Track A)
- **Model**: 3D U-Net (tiny/small/base configs), ISLES26Model with FiLM injection
- **Conditioning**: FiLMConditioner (Track A) + LLMConditioner (Track C)
- **Loss**: ISLES26Loss with Dice + CE + boundary focal
- **Training**: 5-fold CV, PolyLRScheduler, mixed precision, early stopping
- **Evaluation**:Dice, PR-AUC, lesion-wise F1, volume difference

## Diagram Requirements (from SKILL.md)

### Style Requirements
- White/light grey background (#FFFFFF / #F8F8F8)
- Color scheme (LNCS-safe, prints well in greyscale):
  - Data/IO boxes: #E8F4FD (light blue)
  - Model architecture boxes: #EBF5EB (light green)
  - Conditioning gate boxes: #FFF3E0 (light orange) — highlight the novelty
  - Loss/training boxes: #F3E5F5 (light purple)
  - Evaluation boxes: #FFF8E1 (light yellow)
  - Track A (FiLM): #FF8C00 accent
  - Track C (LLM): #6A0DAD accent
- Font: Helvetica or Arial, 8-9pt for labels, 7pt for annotations
- Arrows: orthogonal routing, 1.5pt weight
- No drop shadows (they don't print well at small sizes)
- Dashed border on the conditioning gate box
- Bold border on the U-Net block
- Annotation: "(γ, β)" on the FiLM injection arrow

### Size Constraints (LNCS compliance)
- Full-width (double col): 170mm × 90mm
- Half-width (single col): 83mm × 80mm
