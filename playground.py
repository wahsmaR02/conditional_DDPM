import os, glob, numpy as np, SimpleITK as sitk, matplotlib.pyplot as plt

root = "brain_nii/train"
pid_prefix = "2ABA002"   # change to your patient id prefix

def read_slice(path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
    return arr[0] if arr.ndim == 3 else arr

A = sorted(glob.glob(os.path.join(root, "a", f"{pid_prefix}_z*.nii.gz")))
B = sorted(glob.glob(os.path.join(root, "b", f"{pid_prefix}_z*.nii.gz")))
assert len(A) == len(B) and len(A) > 0, "No slices for that patient"

idx = 0
fig, ax = plt.subplots(1,3,figsize=(10,4))
plt.ion()

def show(i):
    ct, cbct = read_slice(A[i]), read_slice(B[i])
    ax[0].imshow(cbct, cmap="gray"); ax[0].set_title("CBCT"); ax[0].axis("off")
    ax[1].imshow(ct,   cmap="gray"); ax[1].set_title("CT");   ax[1].axis("off")
    ax[2].imshow(cbct, cmap="gray", alpha=0.5); ax[2].imshow(ct, cmap="hot", alpha=0.35)
    ax[2].set_title("Overlay"); ax[2].axis("off")
    fig.suptitle(os.path.basename(A[i]))
    plt.draw(); plt.pause(0.01)

def on_key(event):
    global idx
    if event.key == "right":
        idx = (idx + 1) % len(A)
        show(idx)
    elif event.key == "left":
        idx = (idx - 1) % len(A)
        show(idx)

cid = fig.canvas.mpl_connect('key_press_event', on_key)
show(idx)
plt.show(block=True)
