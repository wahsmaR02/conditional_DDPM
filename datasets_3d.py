# datasets_3d.py
#
# 3D volume dataset for SynthRAD Task 2
# - Loads full cbct.mha / ct.mha / mask.mha
# - Does an 80/20 patient-wise split into train / val
# - On each __getitem__, samples a random 3D patch whose center voxel lies inside the mask

import os
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


# ==============================
# Helper functions
# ==============================

def norm_hu(arr: np.ndarray, lo: float = -1000, hi: float = 2000) -> np.ndarray:
    """
    Clip Hounsfield units to [lo, hi] and scale to [-1,1].
    -1000 ~ air, up to ~2000 ~ dense bone.
    """
    arr = np.clip(arr, lo, hi)
    return (2.0 * (arr - lo) / (hi - lo) - 1.0).astype(np.float32)


def collect_patients(root: str,
                     cohorts: Tuple[str, ...] = ("HN", "TH", "AB")) -> List[Dict]:
    """
    Find all patient folders under root/<cohort>/*.
    Returns a list of dicts:
      {
        "cohort": "HN" or "TH" or "AB",
        "pid":   "patientID",
        "path":  "full/path/to/patient/folder"
      }
    Each patient folder is expected to contain:
      cbct.mha, ct.mha, mask.mha
    """
    patients = []
    for cohort in cohorts:
        cohort_dir = os.path.join(root, cohort)
        if not os.path.isdir(cohort_dir):
            continue

        for p in sorted(os.listdir(cohort_dir)):
            p_dir = os.path.join(cohort_dir, p)
            if not os.path.isdir(p_dir):
                continue

            cbct_path = os.path.join(p_dir, "cbct.mha")
            ct_path   = os.path.join(p_dir, "ct.mha")
            mask_path = os.path.join(p_dir, "mask.mha")

            if not (os.path.exists(cbct_path) and os.path.exists(ct_path) and os.path.exists(mask_path)):
                # skip incomplete patients
                continue

            patients.append({
                "cohort": cohort,
                "pid": p,
                "path": p_dir,
                "cbct_path": cbct_path,
                "ct_path": ct_path,
                "mask_path": mask_path,
            })

    return patients


def split_patients_train_val(patients: List[Dict],
                             train_frac: float = 0.8,
                             seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Patient-wise random split into train / val.
    No test set here (per your instructions).
    """
    rng = random.Random(seed)
    patients_shuffled = patients.copy()
    rng.shuffle(patients_shuffled)

    N = len(patients_shuffled)
    n_train = int(round(train_frac * N))
    n_train = min(max(n_train, 1), N - 1)  # ensure at least 1 train and 1 val

    train_pat = patients_shuffled[:n_train]
    val_pat   = patients_shuffled[n_train:]

    return train_pat, val_pat


# ==============================
# 3D Patch Dataset
# ==============================

class VolumePatchDataset3D(Dataset):
    """
    3D dataset that:
      - loads full cbct.mha / ct.mha / mask.mha volumes
      - samples random 3D patches (D,H,W) where the patch center lies inside the mask

    Returns:
      {
        "CBCT": [1, pD, pH, pW] tensor in [-1,1],
        "pCT":  [1, pD, pH, pW] tensor in [-1,1],
        "meta": dict with cohort, pid, etc. (optional, useful for debugging)
      }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (96, 128, 128),  # (D, H, W)
        cohorts: Tuple[str, ...] = ("HN", "TH", "AB"),
        train_frac: float = 0.8,
        seed: int = 42,
        max_tries: int = 50,
        patches_per_patient: int = 1,
        normalize_hu: bool = True,
    ):
        """
        Args:
            root: Root folder with HN/TH/AB subfolders (SynthRAD Task2 root).
            split: "train" or "val".
            patch_size: (pD, pH, pW) size of 3D patches (in voxels).
            cohorts: Which cohort subfolders to use.
            train_frac: Fraction of patients to use for training (rest for val).
            seed: Random seed (used for patient-split and patch sampling).
            max_tries: Max attempts to find a patch whose center is in the mask.
            patches_per_patient: How many different random patches each patient yields
                                 per epoch (dataset length = n_patients * patches_per_patient).
            normalize_hu: Apply HU → [-1,1] normalization to CBCT/CT.
        """
        super().__init__()
        assert split in ("train", "val"), "split must be 'train' or 'val'"

        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.max_tries = max_tries
        self.patches_per_patient = max(1, patches_per_patient)
        self.normalize_hu = normalize_hu

        # Cache for full volumes so we don't reload from disk each time
        self._cache_cbct = {}
        self._cache_ct = {}
        self._cache_mask = {}


        # Fixed RNG for reproducibility of patch locations
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(seed)

        # 1) Collect all patients
        all_patients = collect_patients(root, cohorts=cohorts)
        if len(all_patients) == 0:
            raise RuntimeError(f"No valid patients found under {root} (cohorts={cohorts})")

        # 2) Train/val split (patient-wise)
        train_pat, val_pat = split_patients_train_val(all_patients, train_frac=train_frac, seed=seed)
        if split == "train":
            self.patients = train_pat
        else:
            self.patients = val_pat

        if len(self.patients) == 0:
            raise RuntimeError(f"No patients in {split} split. Check train_frac and data paths.")

        self.n_patients = len(self.patients)
        # We conceptually repeat patients patches_per_patient times
        self._length = self.n_patients * self.patches_per_patient

        print(f"[VolumePatchDataset3D] root={root}")
        print(f"  Split: {split}")
        print(f"  Patients in split: {self.n_patients}")
        print(f"  patches_per_patient: {self.patches_per_patient}")
        print(f"  Effective dataset length: {self._length}")

    def __len__(self) -> int:
        return self._length

    # ------------------------------
    # Internal helpers
    # ------------------------------

    def _load_volume(self, path: str, is_hu: bool = True) -> np.ndarray:
        """
        Load 3D volume with caching.
        """
        if path in self._cache_ct:
            arr = self._cache_ct[path]
        elif path in self._cache_cbct:
            arr = self._cache_cbct[path]
        else:
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            # Store in correct cache (use filename type to detect)
            if "cbct" in path.lower():
                self._cache_cbct[path] = arr
            else:
                self._cache_ct[path] = arr

        if is_hu and self.normalize_hu:
            arr = norm_hu(arr)

        return arr

    def _load_mask(self, path: str) -> np.ndarray:
        """
        Load mask with caching.
        """
        if path in self._cache_mask:
            return self._cache_mask[path]

        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.uint8)
        self._cache_mask[path] = arr
        return arr


    def _sample_random_corner(self, vol_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        vol_shape = (D, H, W), patch_size = (pD, pH, pW)
        Returns a random valid (z0, y0, x0) s.t. the entire patch fits in the volume.
        """
        D, H, W = vol_shape
        pD, pH, pW = self.patch_size

        if D < pD or H < pH or W < pW:
            raise ValueError(
                f"Volume smaller than patch size: vol={vol_shape}, patch={self.patch_size}. "
                f"Reduce patch_size or pre-resample volumes."
            )

        z0_max = D - pD
        y0_max = H - pH
        x0_max = W - pW

        # randint(high) returns [0, high-1], so we use high+1.
        z0 = int(torch.randint(0, z0_max + 1, (1,), generator=self._torch_rng).item())
        y0 = int(torch.randint(0, y0_max + 1, (1,), generator=self._torch_rng).item())
        x0 = int(torch.randint(0, x0_max + 1, (1,), generator=self._torch_rng).item())

        return z0, y0, x0

    def _sample_valid_patch_corner(self, mask: np.ndarray) -> Tuple[int, int, int]:
        """
        mask: numpy array [D, H, W] with 0/1 (or 0/255).
        Repeats:
          - sample random corner
          - check center voxel of patch is inside mask
        until success or max_tries is reached.
        """
        D, H, W = mask.shape
        pD, pH, pW = self.patch_size

        if D < pD or H < pH or W < pW:
            raise ValueError(
                f"Mask volume smaller than patch size: vol={mask.shape}, patch={self.patch_size}. "
                f"Reduce patch_size or pre-resample volumes."
            )

        for _ in range(self.max_tries):
            z0, y0, x0 = self._sample_random_corner((D, H, W))

            # Center voxel of the patch
            zc = z0 + pD // 2
            yc = y0 + pH // 2
            xc = x0 + pW // 2

            if mask[zc, yc, xc] > 0:
                return z0, y0, x0

        # Fallback: central patch of the volume (even if mask is sparse)
        print("⚠️  VolumePatchDataset3D: could not find valid patch center in mask, "
              "falling back to central patch.")
        z0 = max(0, (D - pD) // 2)
        y0 = max(0, (H - pH) // 2)
        x0 = max(0, (W - pW) // 2)
        return z0, y0, x0

    # ------------------------------
    # Main access
    # ------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns one random 3D patch:
          CBCT: [1, pD, pH, pW] (float32, [-1,1])
          pCT:  [1, pD, pH, pW] (float32, [-1,1])
        """
        # Map global index -> patient index
        patient_index = idx % self.n_patients
        pinfo = self.patients[patient_index]

        # Load full volumes [D, H, W]
        cbct_vol = self._load_volume(pinfo["cbct_path"], is_hu=True)
        ct_vol   = self._load_volume(pinfo["ct_path"],   is_hu=True)
        mask_vol = self._load_mask(pinfo["mask_path"])

        assert cbct_vol.shape == ct_vol.shape == mask_vol.shape, \
            f"Shape mismatch for patient {pinfo['cohort']}/{pinfo['pid']}: " \
            f"CBCT {cbct_vol.shape}, CT {ct_vol.shape}, MASK {mask_vol.shape}"

        # Sample a valid patch corner (center inside mask)
        z0, y0, x0 = self._sample_valid_patch_corner(mask_vol)
        pD, pH, pW = self.patch_size

        # Crop 3D patch
        cbct_patch = cbct_vol[z0:z0 + pD, y0:y0 + pH, x0:x0 + pW]
        ct_patch   = ct_vol[  z0:z0 + pD, y0:y0 + pH, x0:x0 + pW]

        # Convert to torch tensors with channel dimension [1, D, H, W]
        cbct_tensor = torch.from_numpy(cbct_patch).unsqueeze(0)  # [1, D, H, W]
        ct_tensor   = torch.from_numpy(ct_patch).unsqueeze(0)    # [1, D, H, W]

        return {
            "CBCT": cbct_tensor.float(),
            "pCT":  ct_tensor.float(),
            "meta": {
                "cohort": pinfo["cohort"],
                "pid": pinfo["pid"],
            }
        }
