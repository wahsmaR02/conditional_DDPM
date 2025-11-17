# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import glob
import random
import os
import numpy as np #V
import torch as torch #V

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import SimpleITK as sitk

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/a" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/b" % mode) + "/*.*"))

    def __getitem__(self, index):

        image_A = np.load(self.files_A[index % len(self.files_A)],allow_pickle=True)
        image_B = np.load(self.files_B[index % len(self.files_B)],allow_pickle=True)

        item_A = torch.from_numpy(image_A)
        item_B = torch.from_numpy(image_B)
        
        item_A = torch.unsqueeze(item_A,0)
        item_B = torch.unsqueeze(item_B,0)
        return {"a": item_A, "b": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

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
        assert len(self.files_A) == len(self.files_B) > 0, f"No pairs found in {root}/{split}"

        # Ensure filenames match 1–1
        a_base = [os.path.basename(f) for f in self.files_A]
        b_base = [os.path.basename(f) for f in self.files_B]
        assert a_base == b_base, "CT and CBCT filenames do not match"

    def __len__(self):
        return len(self.files_A)

    def __getitem__(self, idx):
        ct_path = self.files_A[idx]
        cbct_path = self.files_B[idx]

        # Read .nii.gz slice as float32 array
        ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
        cbct = sitk.GetArrayFromImage(sitk.ReadImage(cbct_path)).astype(np.float32)

        # Each file is (1,H,W) -> squeeze and re-add channel
        if ct.ndim == 3 and ct.shape[0] == 1:
            ct = ct[0]
        if cbct.ndim == 3 and cbct.shape[0] == 1:
            cbct = cbct[0]

        ct_tensor = torch.from_numpy(ct).unsqueeze(0)    # [1,H,W]
        cbct_tensor = torch.from_numpy(cbct).unsqueeze(0)  # [1,H,W]
        return {"pCT": ct_tensor, "CBCT": cbct_tensor}
