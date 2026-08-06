**Quick answers first:**

**ATLAS versions** — report results on v3.0 only (N=1453). v2.0/v2.1 were guided development; the challenge evaluates on the full v3.0 set. No need to report separately per version.

**Replacing the two fixed cases** — in your Kaggle ingestion notebook:
```python
import shutil, nibabel as nib
from pathlib import Path

# After mounting the SWITCHdrive download as a Kaggle dataset
FIX_DIR  = Path("/kaggle/input/isles26-fixes")   # mount the fix files here
OUT_IMG  = Path("/kaggle/working/atlas_raw/images")
OUT_MASK = Path("/kaggle/working/atlas_raw/masks")

fixes = {
    # sub-r032s056: new T1w provided (over-skull-stripped), no new mask
    "R032__sub-r032s056__ses-1": {"img": FIX_DIR / "sub-r032s056_ses-1_space-orig_desc-brain_T1w.nii.gz", "mask": None},
    # sub-r032s027: file error fixed
    "R032__sub-r032s027__ses-1": {"img": None, "mask": FIX_DIR / "sub-r032s027_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"},
}

for uid, paths in fixes.items():
    if paths["img"]:
        shutil.copy2(paths["img"], OUT_IMG / f"{uid}_T1w.nii.gz")
        print(f"Replaced T1w: {uid}")
    if paths["mask"]:
        shutil.copy2(paths["mask"], OUT_MASK / f"{uid}_rater1.nii.gz")
        print(f"Replaced mask: {uid}")
```
Then re-run preprocessing on only those two UIDs with `overwrite=True`.

---

Now the updated `evaluate.py` integrating the official metrics exactly:Now update the project structure to reflect the `utils/` folder requirement:Three things to action on your end:

**1. Add `utils/` to your repo** — create `utils/__init__.py` (empty) and copy `eval_utils.py` from the organizer's repo verbatim into it. Do not modify it.

**2. Update `requirements.txt`** — add:
```
panoptica>=0.9.0
scikit-image==0.23.2
```

**3. Update `config.yaml`** — fix the metadata dim following the CHRONICITY clarification:
```yaml
conditioning:
  film:
    metadata_dim: 5    # was 4; now [days_norm, is_acute, is_subacute, is_chronic, confirmed_chronic_flag]
```

And update `test_pipeline.py` — change all `torch.rand(B, 4)` meta_vec fixtures to `torch.rand(B, 5)` so smoke tests don't break on the shape change.

**On the Docker deadline** — the 10-minute T4 limit with Track C (LLM forward pass per scan) is a real risk. Once fold 0 trains, time a single inference pass before committing to Track C as your submission model.