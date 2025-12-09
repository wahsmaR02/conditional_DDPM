# datasets_3d.py
#
# 3D volume dataset for SynthRAD Task 2
# - Loads full cbct.mha / ct.mha / mask.mha
# - Performs a 70/15/15 patient split into train/val/test.
# - Samples valid 3D patches where the patch center lies inside the mask.

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
    """
    arr = np.clip(arr, lo, hi)
    return (2.0 * (arr - lo) / (hi - lo) - 1.0).astype(np.float32)


def collect_patients(root: str,
                     cohorts: Tuple[str, ...] = ("HN", "TH", "AB")) -> List[Dict]:
    """
    Find all patient folders under root/<cohort>/*.
    Returns entries with paths to cbct.mha, ct.mha, mask.mha.
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


def split_patients_train_val_test(
    patients: List[Dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Deterministic patient-wise split into train/val/test.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = random.Random(seed)
    patients_shuffled = patients.copy()
    rng.shuffle(patients_shuffled)

    N = len(patients_shuffled)
    n_train = int(round(train_frac * N))
    n_val   = int(round(val_frac * N))

    # Safety constraints
    n_train = max(1, min(n_train, N - 2))
    n_val   = max(1, min(n_val, N - n_train - 1))

    train_pat = patients_shuffled[:n_train]
    val_pat   = patients_shuffled[n_train:n_train + n_val]
    test_pat  = patients_shuffled[n_train + n_val:]

    return train_pat, val_pat, test_pat


# ==============================
# VolumePatchDataset3D
# ==============================

class VolumePatchDataset3D(Dataset):
    """
    Loads full SynthRAD Task2 volumes (CBCT/CT/mask),
    and returns random 3D patches with center inside the mask.

    __getitem__ returns:
        {
            "CBCT": FloatTensor [1, D, H, W],
            "pCT":  FloatTensor [1, D, H, W],
            "meta": {"cohort", "pid"}
        }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (96, 128, 128),
        cohorts: Tuple[str, ...] = ("HN", "TH", "AB"),
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        max_tries: int = 50,
        patches_per_patient: int = 1,
        normalize_hu: bool = True,
    ):
        super().__init__()
        assert split in ("train", "val", "test")

        self.root = root
        self.split = split
        self.patch_size = patch_size
        self.max_tries = max_tries
        self.patches_per_patient = max(1, patches_per_patient)
        self.normalize_hu = normalize_hu

        # Caches for full volumes
        self._cache_cbct = {}
        self._cache_ct = {}
        self._cache_mask = {}

        # RNG for patch sampling
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(seed)

        # -------------------------------
        # 1) Collect all patients
        # -------------------------------
        all_patients = collect_patients(root, cohorts=cohorts)
        if len(all_patients) == 0:
            raise RuntimeError(f"No patients found under {root}")

        # -------------------------------
        # 2) Train/val/test split
        # -------------------------------
        train_pat, val_pat, test_pat = split_patients_train_val_test(
            all_patients,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed
        )

        if split == "train":
            self.patients = train_pat
        elif split == "val":
            self.patients = val_pat
        else:
            self.patients = test_pat

        if len(self.patients) == 0:
            raise RuntimeError(f"Split '{split}' has 0 patients! Check split fractions.")

        self.n_patients = len(self.patients)
        self._length = self.n_patients * self.patches_per_patient

        print(f"[VolumePatchDataset3D] Split={split} | Patients={self.n_patients} | Length={self._length}")

    def __len__(self):
        return self._length

    # ------------------------------
    # Loading volumes with caching
    # ------------------------------

    def _load_volume(self, path: str, is_hu: bool = True) -> np.ndarray:
        """
        Load volume with caching.
        """
        if path.endswith("cbct.mha"):
            cache = self._cache_cbct
        else:
            cache = self._cache_ct

        if path not in cache:
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            if is_hu and self.normalize_hu:
                arr = norm_hu(arr)
            cache[path] = arr

        return cache[path]

    def _load_mask(self, path: str) -> np.ndarray:
        if path not in self._cache_mask:
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img).astype(np.uint8)
            self._cache_mask[path] = arr
        return self._cache_mask[path]

    # ------------------------------
    # Patch sampling helpers
    # ------------------------------

    def _sample_random_corner(self, vol_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        D, H, W = vol_shape
        pD, pH, pW = self.patch_size

        z0 = int(torch.randint(0, D - pD + 1, (1,), generator=self._torch_rng))
        y0 = int(torch.randint(0, H - pH + 1, (1,), generator=self._torch_rng))
        x0 = int(torch.randint(0, W - pW + 1, (1,), generator=self._torch_rng))
        return z0, y0, x0

    def _sample_valid_patch_corner(self, mask: np.ndarray) -> Tuple[int, int, int]:
        D, H, W = mask.shape
        pD, pH, pW = self.patch_size

        for _ in range(self.max_tries):
            z0, y0, x0 = self._sample_random_corner(mask.shape)
            zc = z0 + pD // 2
            yc = y0 + pH // 2
            xc = x0 + pW // 2
            if mask[zc, yc, xc] > 0:
                return z0, y0, x0

        # fallback
        print("⚠️  Could not find valid patch inside mask, using central patch")
        return (
            max(0, (D - pD) // 2),
            max(0, (H - pH) // 2),
            max(0, (W - pW) // 2)
        )

    # ------------------------------
    # __getitem__
    # ------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        patient_index = idx % self.n_patients
        pinfo = self.patients[patient_index]

        cbct = self._load_volume(pinfo["cbct_path"], is_hu=True)
        ct   = self._load_volume(pinfo["ct_path"],   is_hu=True)
        mask = self._load_mask(pinfo["mask_path"])

        assert cbct.shape == ct.shape == mask.shape

        z0, y0, x0 = self._sample_valid_patch_corner(mask)
        pD, pH, pW = self.patch_size

        cbct_patch = cbct[z0:z0+pD, y0:y0+pH, x0:x0+pW]
        ct_patch   = ct[z0:z0+pD, y0:y0+pH, x0:x0+pW]

        return {
            "CBCT": torch.from_numpy(cbct_patch).unsqueeze(0),
            "pCT":  torch.from_numpy(ct_patch).unsqueeze(0),
            "meta": {"cohort": pinfo["cohort"], "pid": pinfo["pid"]}
        }
