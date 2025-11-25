# import os, glob, argparse
# import numpy as np
# import SimpleITK as sitk

# def norm_hu(arr, lo=-1000, hi=2000):
#     arr = np.clip(arr, lo, hi)
#     return (2.0 * (arr - lo) / (hi - lo) - 1.0).astype(np.float32)

# def pad_or_crop_to(arr, h=256, w=256):
#     H, W = arr.shape
#     y0 = max(0, (H - h)//2); y1 = y0 + min(H, h)
#     x0 = max(0, (W - w)//2); x1 = x0 + min(W, w)
#     cropped = arr[y0:y1, x0:x1]
#     ph = max(0, h - cropped.shape[0]); pw = max(0, w - cropped.shape[1])
#     top = ph//2; bottom = ph - top; left = pw//2; right = pw - left
#     if ph>0 or pw>0:
#         cropped = np.pad(cropped, ((top,bottom),(left,right)), mode="edge")
#     return cropped.astype(np.float32)

# def export_to_nii(root, out, split="train", cohorts=("HN","TH","AB"), size=256):
#     os.makedirs(os.path.join(out, split, "a"), exist_ok=True)  # CT
#     os.makedirs(os.path.join(out, split, "b"), exist_ok=True)  # CBCT

#     for cohort in cohorts:
#         patients = sorted(glob.glob(os.path.join(root, cohort, "*")))
#         for p in patients:
#             pid = os.path.basename(p)
#             cbct_path = os.path.join(p, "cbct.mha")
#             ct_path   = os.path.join(p, "ct.mha")
#             if not (os.path.exists(cbct_path) and os.path.exists(ct_path)):
#                 continue

#             cbct = sitk.GetArrayFromImage(sitk.ReadImage(cbct_path)).astype(np.float32)
#             ct   = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
#             cbct = norm_hu(cbct); ct = norm_hu(ct)

#             Z = cbct.shape[0]
#             for z in range(Z):
#                 cbct_z = pad_or_crop_to(cbct[z], size, size)
#                 ct_z   = pad_or_crop_to(ct[z], size, size)

#                 cbct_img = sitk.GetImageFromArray(cbct_z)
#                 ct_img   = sitk.GetImageFromArray(ct_z)
#                 sitk.WriteImage(cbct_img, os.path.join(out, split, "b", f"{pid}_z{z:03d}.nii.gz"))
#                 sitk.WriteImage(ct_img,   os.path.join(out, split, "a", f"{pid}_z{z:03d}.nii.gz"))
#     print(f"Slices saved as .nii.gz under {out}/{split}/a,b")

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--root", required=True, help="Root folder with HN/TH/AB subfolders")
#     ap.add_argument("--out",  default="./brain_nii", help="Output folder for .nii.gz slices")
#     ap.add_argument("--split", default="train")
#     ap.add_argument("--size", type=int, default=256)
#     args = ap.parse_args()

#     export_to_nii(args.root, args.out, args.split, size=args.size)

# test for push

import os, glob, argparse
import numpy as np
import SimpleITK as sitk

def norm_hu(arr, lo=-1000, hi=2000): # Hounsfeld unit -1000 is air, +2000 is dense bone
    arr = np.clip(arr, lo, hi) # remove noise spike, reconstruction artifacts.
    return (2.0 * (arr - lo) / (hi - lo) - 1.0).astype(np.float32) # normalize to [-1,1]

def pad_or_crop_to(arr, h=256, w=256):
    H, W = arr.shape
    y0 = max(0, (H - h)//2); y1 = y0 + min(H, h)
    x0 = max(0, (W - w)//2); x1 = x0 + min(W, w)
    cropped = arr[y0:y1, x0:x1]
    ph = max(0, h - cropped.shape[0]); pw = max(0, w - cropped.shape[1])
    top = ph//2; bottom = ph - top; left = pw//2; right = pw - left
    if ph>0 or pw>0:
        cropped = np.pad(cropped, ((top,bottom),(left,right)), mode="edge")
    return cropped.astype(np.float32)

def crop_with_mask(img, mask):
    """Crop img to the bounding box of mask (nonzero region)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # empty mask → return original
        return img

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    return img[ymin:ymax+1, xmin:xmax+1]


def export_to_nii(root, out, split="train", cohorts=("HN","TH","AB"), size=256):
    os.makedirs(os.path.join(out, split, "a"), exist_ok=True)  # CT
    os.makedirs(os.path.join(out, split, "b"), exist_ok=True)  # CBCT

    for cohort in cohorts:
        patients = sorted(glob.glob(os.path.join(root, cohort, "*")))
        for p in patients:
            pid = os.path.basename(p)
            cbct_path = os.path.join(p, "cbct.mha")
            ct_path   = os.path.join(p, "ct.mha")
            mask_path = os.path.join(p, "mask.mha")  # ADD MASK

            if not (os.path.exists(cbct_path) and os.path.exists(ct_path) and os.path.exists(mask_path)):
                continue

            # Load volumes
            cbct = sitk.GetArrayFromImage(sitk.ReadImage(cbct_path)).astype(np.float32)
            ct   = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path)).astype(np.uint8)

            # Normalize CT/CBCT
            cbct = norm_hu(cbct)
            ct   = norm_hu(ct)

            skip = 5  # skip slices at beginning and end
            Z = cbct.shape[0]  # number of slices
            for z in range(skip+1, Z-skip):

                slice_cbct = cbct[z]
                slice_ct   = ct[z]
                slice_mask = mask[z]

                # Crop using mask
                slice_cbct = crop_with_mask(slice_cbct, slice_mask)
                slice_ct   = crop_with_mask(slice_ct, slice_mask)

                # Then pad/crop to target size
                slice_cbct = pad_or_crop_to(slice_cbct, size, size)
                slice_ct   = pad_or_crop_to(slice_ct, size, size)

                # Convert back to SimpleITK images
                cbct_img = sitk.GetImageFromArray(slice_cbct)
                ct_img   = sitk.GetImageFromArray(slice_ct)

                sitk.WriteImage(cbct_img, os.path.join(out, split, "b", f"{pid}_z{z:03d}.nii.gz"))
                sitk.WriteImage(ct_img,   os.path.join(out, split, "a", f"{pid}_z{z:03d}.nii.gz"))

    print(f"Slices saved as .nii.gz under {out}/{split}/a,b")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder with HN/TH/AB subfolders")
    ap.add_argument("--out",  default="./brain_nii", help="Output folder for .nii.gz slices")
    ap.add_argument("--split", default="train")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    export_to_nii(args.root, args.out, args.split, size=args.size)
