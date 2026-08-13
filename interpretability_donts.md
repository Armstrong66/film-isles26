# Coding Assistant Instructions — Interpretability / Docker Isolation

## The rule
`interpretability.py` and all visualization hooks must NEVER be imported or
executed inside the Docker inference entrypoint. The Docker container runs only:
  preprocessing → model.forward() → post-processing → save mask

## Required changes (implement these exactly)

### 1. Keep hooks permanently opt-in in model.py
The FiLM and embedding hooks must only fire when explicitly attached.
This is already correct IF the guard is:

    if getattr(self, '_film_hook', None) is not None:
        self._film_hook(...)

NOT a bare `hasattr` — use `getattr(..., None)` so unset attributes
return None rather than raising AttributeError. Docker never sets these
attributes, so the check costs ~0ns and the hook never fires.

### 2. Guard the --save-probs flag in evaluate.py
Saving prob maps must be gated by a CLI flag, never on by default:

    parser.add_argument("--save-probs", action="store_true")
    # Only runs locally, never in Docker
    if args.save_probs:
        np.save(prob_dir / f"{uid}_prob.npy", soft_map)

### 3. interpretability.py is NEVER imported in entrypoint.py
The Docker entrypoint must import ONLY:

    from preprocessing import reorient_to_ras, clip_and_normalise
    from model import build_model, ISLES26Model
    from train import load_checkpoint

No import of interpretability, visualize, captum, umap, or matplotlib.
Add this comment at the top of entrypoint.py:

    # INTERPRETABILITY IMPORTS ARE EXPLICITLY EXCLUDED FROM THIS FILE.
    # See pipeline/interpretability.py for post-hoc analysis (local only).

### 4. requirements-docker.txt (separate from requirements.txt)
Create a minimal requirements file for the Docker image:

    torch==2.3.1
    nibabel==5.2.1
    monai==1.3.2
    omegaconf==2.3.0
    sentence-transformers==3.0.1   # Track C only; remove for Track A Docker

DO NOT include in requirements-docker.txt:
    captum, umap-learn, matplotlib, seaborn, scikit-image,
    panoptica, scipy (inference doesn't need it), pandas, jupyter*

### 5. Dockerfile — copy only inference files
    COPY pipeline/preprocessing.py  /opt/algorithm/
    COPY pipeline/conditioning.py   /opt/algorithm/
    COPY pipeline/model.py          /opt/algorithm/
    COPY pipeline/augmentation.py   /opt/algorithm/
    COPY pipeline/entrypoint.py     /opt/algorithm/
    COPY configs/config.yaml        /opt/algorithm/
    COPY checkpoints/               /opt/algorithm/checkpoints/
    # interpretability.py is NOT copied

### 6. Verification — run before Docker submission
    python -c "
    import sys
    sys.path.insert(0, 'pipeline')
    import entrypoint
    forbidden = ['captum','umap','matplotlib','seaborn','interpretability']
    for mod in forbidden:
        assert mod not in sys.modules, f'FAIL: {mod} was imported by entrypoint'
    print('OK: no interpretability deps in inference path')
    "