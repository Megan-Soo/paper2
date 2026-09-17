#!/usr/bin/env python3
import numpy as np
import SimpleITK as sitk
import os, argparse, re

def natural_key(s):
    """
    Split string into list of text and integer chunks for natural sorting.
    e.g. 'img10.nii.gz' -> ['img', 10, '.nii.gz']
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read directory containing mask NifTis, export volumes in .npz")
    parser.add_argument("dir", type=str, help="Dir containing mask NiFTis")
    parser.add_argument("--o", type=str, required=False, help="Path to save array of lung mask volumes.")
    args = parser.parse_args()
    dir = args.dir
    out = args.o

    # Read lung masks of phases
    image_files = sorted(
        [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(".nii.gz")],
        key=natural_key
    )

    vols = [] # Stack into np array
    for i,f in enumerate(image_files):
        img = sitk.ReadImage(f)
        spacing = img.GetSpacing()
        img = sitk.GetArrayFromImage(img)
        vols.append(np.count_nonzero(img) * np.prod(spacing))
        print(f'Phase {i+1} {np.count_nonzero(img) * np.prod(spacing)/10**6:.2f}')
        r = np.count_nonzero(img==1) * np.prod(spacing)
        l = np.count_nonzero(img==2) * np.prod(spacing)
        print(f"Left lung {l/10**6:.2f}; Right lung {r/10**6:.2f}")

    vols = np.array(vols)
    if out:
        np.savez_compressed(out,vols)