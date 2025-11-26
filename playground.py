import torch
from datasets_3d import VolumePatchDataset3D   # your file
import numpy as np

ROOT = "synthRAD2025_Task2_Train/playground"

def main():

    print("Loading dataset…")
    dataset = VolumePatchDataset3D(
        root=ROOT,
        split="train",
        patch_size=(64, 128, 128),  # (D,H,W)
        seed=42,
    )

    print("Dataset loaded")
    print("Number of patients:", len(dataset.patients))

    # ------- Test 1: Try reading one item -------
    print("\nSampling one patch…")
    sample = dataset[0]
    ct = sample["pCT"]
    cbct = sample["CBCT"]

    print("Patch shapes:")
    print("  pCT :", ct.shape)
    print("  CBCT:", cbct.shape)

    # Should be [1, D, H, W]
    assert ct.ndim == 4 and cbct.ndim == 4
    assert ct.shape == cbct.shape

    print("\nValue ranges:")
    print("  CT  :", float(ct.min()), "to", float(ct.max()))
    print("  CBCT:", float(cbct.min()), "to", float(cbct.max()))

    # ------- Test 2: Draw several random patches -------
    print("\nDrawing 5 random patches to test mask-center validity…")
    for i in range(5):
        patch = dataset[np.random.randint(len(dataset))]
        ct = patch["pCT"]
        center_val = ct[0, ct.shape[1]//2, ct.shape[2]//2, ct.shape[3]//2].item()
        print(f" Patch {i}: center voxel =", center_val)

    print("\nIf all center voxels are > -1, then mask constraint works.")

if __name__ == "__main__":
    main()
    