# drawio-skill Setup for ISLES26

## Installation Status

✅ drawio-skill cloned to: `C:/Users/DELL/.claude/skills/drawio-skill/`

## Current Limitation

❌ draw.io desktop CLI is **not available** on this Windows machine.

The draw.io app was installed from the Microsoft Store as a UWP app (`draw.io.draw.ioDiagrams` version 31.1.8.0).
UWP apps do not expose a CLI binary that can be invoked from the command line.

## Alternative Solutions

### Option A: Browser Fallback (Recommended - No install needed)

Use the `encode_drawio_url.py` script to generate shareable URLs:

```bash
# Generate a viewer URL (read-only)
python3 "C:/Users/DELL/.claude/skills/drawio-skill/skills/drawio-skill/scripts/encode_drawio_url.py" diagram.drawio

# Generate an editor URL (editable)
python3 "C:/Users/DELL/.claude/skills/drawio-skill/skills/drawio-skill/scripts/encode_drawio_url.py" --edit diagram.drawio
```

### Option B: Manual diagrams.net Web UI

1. Open https://app.diagrams.net
2. File → Import From → Device → paste XML
3. Export → PNG at 600 DPI

### Option C: Install Standalone draw.io

Download and install the desktop version from:
https://github.com/jgraph/drawio-desktop/releases

After installation, the CLI will be available at:
- `C:\Program Files\draw.io\draw.io.exe`

## Python Fallback Scripts

The skill includes Python scripts for:
- `encode_drawio_url.py` - generates diagrams.net URLs (no CLI needed)
- `validate.py` - structural linting
- Various importers (pyimports, c4, autolayout, etc.)

Python ≥ 3.8 is required for these scripts.

## ISLES26 Diagram Configuration

### Target Output Directory
```
C:\Users\DELL\Downloads\film-isles26\figures\
```

### Diagram Requirements
- Format: `.drawio` (XML) or `.drawio.png` (exported)
- Style: LNCS-compliant (prints well in greyscale)
- Size: ≤ 170mm × 90mm (double-column LNCS)

### Diagram Types
1. **Full methodology pipeline** - Figure 1
2. **Conditioning module detail** - Figure 2
3. **Results overview** - Results section

## Next Steps

1. **Use browser fallback** to generate and export diagrams:
   - Generate `.drawio` XML
   - Open in https://app.diagrams.net
   - Export PNG at 600 DPI

2. **Or install standalone draw.io** for CLI automation:
   - Download from https://github.com/jgraph/drawio-desktop/releases
   - Add `C:\Program Files\draw.io\draw.io.exe` to PATH
