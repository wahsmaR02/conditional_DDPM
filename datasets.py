import glob
import random
import os
import numpy as np #V
import torch as torch #V

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import SimpleITK as sitk

# -------------------------------------------------------------------
# New dataset class for .nii.gz slices (preferred for SynthRAD)
# -------------------------------------------------------------------
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
