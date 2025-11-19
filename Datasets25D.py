# Datasets25D.py

import glob
import os
import re
import numpy as np
import torch

from torch.utils.data import Dataset
import SimpleITK as sitk


class ImageDatasetNii25D(Dataset):
    """
    Loads per-slice .nii.gz pairs exported by New_Dataloader.py:
      root/split/a/{cohort}_{pid}_zNNN.nii.gz  (pCT)
      root/split/b/{cohort}_{pid}_zNNN.nii.gz  (CBCT)
      root/split/m/{cohort}_{pid}_zNNN.nii.gz  (mask)

    For each *center* slice we return a stack of K = 2*half_width+1 slices:
      pCT:  [K, H, W]
      CBCT: [K, H, W]
      mask: [1, H, W] (center mask only)
    """
    def __init__(self, root, split="train", half_width=1):
        """
        half_width=1 => K=3 (z-1, z, z+1)
        half_width=2 => K=5 (z-2..z+2)
        """
        self.root = root
        self.split = split
        self.half_width = half_width

        files_A = sorted(glob.glob(os.path.join(root, split, "a", "*.nii.gz")))
        files_B = sorted(glob.glob(os.path.join(root, split, "b", "*.nii.gz")))
        files_m = sorted(glob.glob(os.path.join(root, split, "m", "*.nii.gz")))

        assert len(files_A) == len(files_B) > 0, f"No pairs found in {root}/{split}"

        a_base = [os.path.basename(f) for f in files_A]
        b_base = [os.path.basename(f) for f in files_B]
        assert a_base == b_base, "CT and CBCT filenames do not match"

        # ---- Parse filenames into (base_id, z) and organize by patient ----
        self.samples = []          # flat list of all slices
        self.by_base = {}          # base_id -> {"zs": [z sorted], "by_z": {z: sample_dict}}

        for ct_path, cbct_path, mask_path in zip(files_A, files_B, files_m):
            fname = os.path.basename(ct_path)
            # Expect pattern: something_zNNN.nii.gz
            # Split at '_z'
            if "_z" not in fname:
                raise RuntimeError(f"Filename does not contain '_z': {fname}")
            base_part, z_part = fname.split("_z", 1)
            # z_part ~ '012.nii.gz' -> take '012'
            z_str = z_part.split(".nii")[0]
            z = int(z_str)

            sample = {
                "base": base_part,
                "z": z,
                "ct": ct_path,
                "cbct": cbct_path,
                "mask": mask_path,
            }
            self.samples.append(sample)

            if base_part not in self.by_base:
                self.by_base[base_part] = {"zs": [], "by_z": {}}
            self.by_base[base_part]["zs"].append(z)
            self.by_base[base_part]["by_z"][z] = sample

        # Sort z lists for each patient
        for base, info in self.by_base.items():
            info["zs"] = sorted(info["zs"])

    def __len__(self):
        return len(self.samples)

    def _get_clamped_sample(self, base, z):
        """Get the sample for (base, z) clamped to [z_min, z_max] of that base."""
        info = self.by_base[base]
        zs = info["zs"]
        z_clamped = min(max(z, zs[0]), zs[-1])
        return info["by_z"][z_clamped]

    def __getitem__(self, idx):
        center = self.samples[idx]
        base = center["base"]
        z0 = center["z"]

        ct_slices = []
        cbct_slices = []

        # Build the K-slice stack around the center
        for offset in range(-self.half_width, self.half_width + 1):
            s = self._get_clamped_sample(base, z0 + offset)

            ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(s["ct"])).astype(np.float32)
            cbct_arr = sitk.GetArrayFromImage(sitk.ReadImage(s["cbct"])).astype(np.float32)

            # Each .nii.gz is [1, H, W] -> squeeze to [H, W]
            if ct_arr.ndim == 3 and ct_arr.shape[0] == 1:
                ct_arr = ct_arr[0]
            if cbct_arr.ndim == 3 and cbct_arr.shape[0] == 1:
                cbct_arr = cbct_arr[0]

            ct_slices.append(torch.from_numpy(ct_arr))
            cbct_slices.append(torch.from_numpy(cbct_arr))

        # Stack into [K, H, W]
        ct_tensor = torch.stack(ct_slices, dim=0)
        cbct_tensor = torch.stack(cbct_slices, dim=0)

        # Center mask only
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(center["mask"])).astype(np.float32)
        if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
            mask_arr = mask_arr[0]
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # [1, H, W]

        return {"pCT": ct_tensor, "CBCT": cbct_tensor, "mask": mask_tensor}
