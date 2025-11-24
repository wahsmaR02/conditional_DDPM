import glob
import random
import os
import numpy as np #V
import torch as torch #V

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import SimpleITK as sitk

class ImageDatasetNii3D(Dataset):
    """
    Loads full 3D paired .nii.gz volumes:

      root/split/a/*.nii.gz   (CT)
      root/split/b/*.nii.gz   (CBCT)
      root/split/m/*.nii.gz   (mask, optional)

    Returns tensors:
      CT:   [1, D, H, W]
      CBCT: [1, D, H, W]
      mask: [1, D, H, W]  (if present)
    """

    def __init__(self, root, split="train"):
        self.files_CT   = sorted(glob.glob(os.path.join(root, split, "a", "*.nii.gz")))
        self.files_CBCT = sorted(glob.glob(os.path.join(root, split, "b", "*.nii.gz")))
        self.files_mask = sorted(glob.glob(os.path.join(root, split, "m", "*.nii.gz")))

        assert len(self.files_CT) > 0, f"No CT files in {root}/{split}/a"
        assert len(self.files_CBCT) > 0, f"No CBCT files in {root}/{split}/b"
        assert len(self.files_CT) == len(self.files_CBCT), "CT and CBCT file counts differ"

        # Ensure filenames match 1–1
        ct_base = [os.path.basename(f) for f in self.files_CT]
        cb_base = [os.path.basename(f) for f in self.files_CBCT]
        assert ct_base == cb_base, "CT and CBCT filenames do not match"

        # Check mask count (optional)
        if len(self.files_mask) > 0:
            m_base = [os.path.basename(f) for f in self.files_mask]
            assert m_base == ct_base, "Mask filenames do not match CT filenames"

    def __len__(self):
        return len(self.files_CT)   # one file = one patient

    def load_3d(self, path):
        vol = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(vol).astype(np.float32)   # [D,H,W]
        tensor = torch.from_numpy(arr).unsqueeze(0)            # → [1,D,H,W]
        return tensor

    def __getitem__(self, idx):
        ct   = self.load_3d(self.files_CT[idx])
        cbct = self.load_3d(self.files_CBCT[idx])

        sample = {"CT": ct, "CBCT": cbct}

        if len(self.files_mask) > 0:
            mask = self.load_3d(self.files_mask[idx])
            sample["mask"] = mask

        return sample

class ImageDatasetNii_25D(Dataset):
    """
    Loads 2.5D stacks:
      CT_stack:   [3,H,W]
      CBCT_stack: [3,H,W]
    """

    def __init__(self, root, split="train", k=1):
        self.k = k

        self.files_CT = sorted(glob.glob(os.path.join(root, split, "a", "*.nii.gz")))
        self.files_CBCT = sorted(glob.glob(os.path.join(root, split, "b", "*.nii.gz")))

        assert len(self.files_CT) > 0, f"No CT files in {root}/{split}/a"
        assert len(self.files_CBCT) > 0, f"No CBCT files in {root}/{split}/b"

        # Ensure filenames match 1–1
        ct_base = [os.path.basename(f) for f in self.files_CT]
        cb_base = [os.path.basename(f) for f in self.files_CBCT]
        assert ct_base == cb_base, "CT and CBCT filenames do not match"

        # Group files by patient
        self.by_patient_CT = {}
        self.by_patient_CBCT = {}

        for ct_f, cb_f in zip(self.files_CT, self.files_CBCT):
            base = os.path.basename(ct_f)
            pid, zstr = base.split("_z")
            self.by_patient_CT.setdefault(pid, []).append(ct_f)
            self.by_patient_CBCT.setdefault(pid, []).append(cb_f)

        # Sorting function
        def sort_by_z(lst):
            return sorted(lst, key=lambda f: int(os.path.basename(f).split("_z")[1].split(".")[0]))

        # Sort each patient's slice list
        for pid in self.by_patient_CT:
            self.by_patient_CT[pid] = sort_by_z(self.by_patient_CT[pid])
            self.by_patient_CBCT[pid] = sort_by_z(self.by_patient_CBCT[pid])

    def load_slice(self, path):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]    # (1,H,W) → (H,W)
        return arr

    def __len__(self):
        return len(self.files_CT)

    def __getitem__(self, idx):
        ct_path = self.files_CT[idx]
        base = os.path.basename(ct_path)
        pid, zstr = base.split("_z")
        z = int(zstr.split(".")[0])

        ct_list = self.by_patient_CT[pid]
        cbct_list = self.by_patient_CBCT[pid]
        Z = len(ct_list)

        def clamp(i):
            return max(0, min(Z - 1, i))

        ct_stack = []
        cbct_stack = []

        for dz in [-1, 0, 1]:  # z-1, z, z+1
            zz = clamp(z + dz)
            ct_stack.append(self.load_slice(ct_list[zz]))
            cbct_stack.append(self.load_slice(cbct_list[zz]))

        ct_stack = torch.tensor(np.stack(ct_stack), dtype=torch.float32)
        cbct_stack = torch.tensor(np.stack(cbct_stack), dtype=torch.float32)

        return {
            "CT": ct_stack,
            "CBCT": cbct_stack
        }

class ImageDatasetNii(Dataset):
    """
    Loads per-slice .nii.gz pairs:
      root/split/a/*.nii.gz  (CT)
      root/split/b/*.nii.gz  (CBCT)
    Returns:
      {"pCT": [1,H,W], "CBCT": [1,H,W]} as float32 tensors in [-1,1].
    """
    def __init__(self, root, split="train"):
        self.files_A = sorted(glob.glob(os.path.join(root, split, "a", "*.nii.gz")))
        self.files_B = sorted(glob.glob(os.path.join(root, split, "b", "*.nii.gz")))
        self.files_m = sorted(glob.glob(os.path.join(root, split, "m", "*.nii.gz")))
        assert len(self.files_A) == len(self.files_B) > 0, f"No pairs found in {root}/{split}"

        # Ensure filenames match 1–1
        a_base = [os.path.basename(f) for f in self.files_A]
        b_base = [os.path.basename(f) for f in self.files_B]
        #m_base = [os.path.basename(f) for f in self.files_m]
        assert a_base == b_base, "CT and CBCT filenames do not match"

    def __len__(self):
        return len(self.files_A)

    def __getitem__(self, idx):
        ct_path = self.files_A[idx]
        cbct_path = self.files_B[idx]
        mask_path = self.files_m[idx]

        # Read .nii.gz slice as float32 array
        ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
        cbct = sitk.GetArrayFromImage(sitk.ReadImage(cbct_path)).astype(np.float32)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.float32)

        # Each file is (1,H,W) -> squeeze and re-add channel
        if ct.ndim == 3 and ct.shape[0] == 1:
            ct = ct[0]
        if cbct.ndim == 3 and cbct.shape[0] == 1:
            cbct = cbct[0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        ct_tensor = torch.from_numpy(ct).unsqueeze(0)    # [1,H,W]
        cbct_tensor = torch.from_numpy(cbct).unsqueeze(0)  # [1,H,W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return {"pCT": ct_tensor, "CBCT": cbct_tensor, "mask": mask_tensor}