#!/usr/bin/env python3
import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import os, argparse

def binary_erosion(image, radius=2,out=None):
    """
    Apply binary erosion to a 3D SimpleITK image.

    Parameters:
    - image (sitk.Image): Input 3D volumetric image.
    - radius (int or tuple): Radius of the binary erosion kernel. Can be an integer or a 3D tuple.

    Returns:
    - sitk.Image: Filtered 3D image.
    """
    erosion_filter = sitk.BinaryErodeImageFilter()
    erosion_filter.SetKernelRadius([radius,radius,radius])
    eroded = erosion_filter.Execute(image)
    eroded.CopyInformation(image)
    if out is not None:
        sitk.WriteImage(eroded,out)
    return eroded

def morphological_closing_on_multilabel_mask(mask_img: sitk.Image, radius_voxels: int, kernel_ball=True):
    """
    Apply morphological closing to each nonzero label in a multi-labelled mask.

    Parameters
    ----------
    mask_img : sitk.Image
        Multi-label mask image (0 = background, >0 = label values).
    radius_voxels : int
        Radius of structuring element in voxels.

    Returns
    -------
    sitk.Image
        Closed multi-label mask.
    """
    # Convert to numpy for label iteration
    mask_array = sitk.GetArrayFromImage(mask_img)  # (z,y,x)
    labels = np.unique(mask_array)
    labels = labels[labels != 0]  # exclude background

    closed_array = np.zeros_like(mask_array, dtype=np.uint16)

    for lab in labels:
        # Extract binary mask for this label
        bin_mask = (mask_array == lab).astype(np.uint8)
        bin_img = sitk.GetImageFromArray(bin_mask)
        bin_img.CopyInformation(mask_img)

        # Apply morphological closing
        closing = sitk.BinaryMorphologicalClosingImageFilter()
        closing.SetForegroundValue(1)
        closing.SetKernelRadius([int(radius_voxels)] * mask_img.GetDimension())
        if kernel_ball:
            closing.SetKernelType(sitk.sitkBall) # optional
        closed_bin = closing.Execute(bin_img)

        # Add to combined array
        closed_array[sitk.GetArrayFromImage(closed_bin) > 0] = lab

    closed_img = sitk.GetImageFromArray(closed_array)
    closed_img.CopyInformation(mask_img)
    return closed_img

def space_partitioning(lung_mask: np.ndarray, compact_mask: np.ndarray) -> np.ndarray:
    """
    Perform space partitioning of a lung mask given a labelled compact subset mask.

    Parameters
    ----------
    lung_mask : np.ndarray
        3D binary mask (same shape as compact_mask) where 1/True indicates lung voxels.
    compact_mask : np.ndarray
        3D integer mask (same shape) where 0 = background, 
        and positive integers are distinct labels (compact subsets).

    Returns
    -------
    labelled_lobes : np.ndarray
        3D integer mask with same shape, where each lung voxel is assigned
        to the label of the nearest compact subset.
    """
    if lung_mask.shape != compact_mask.shape:
        raise ValueError("lung_mask and compact_mask must have the same shape.")
    
    # Background is where compact_mask == 0
    background = compact_mask == 0

    # Compute Euclidean distance transform and indices of nearest labeled voxel
    distances, indices = ndimage.distance_transform_edt(
        background,
        return_indices=True
    )

    # Use indices to fetch the nearest label from the compact_mask
    nearest_labels = compact_mask[tuple(indices)]

    # Keep only inside the lung mask, otherwise set to 0
    labelled_lobes = np.where(lung_mask, nearest_labels, 0)

    return labelled_lobes

def main():
    parser = argparse.ArgumentParser(description="Read multilabel vessel mask per lung and dilates into binary lung mask to form lobes mask.")
    parser.add_argument("-mask_vessel", required=True, help="Path to multilabel vessel mask file (e.g. mask.nii.gz)")
    parser.add_argument("-mask_lung", required=True, help="Path to lung mask file (e.g. mask.nii.gz)")
    parser.add_argument("-o", required=False, help="Path to save the lobes mask")
    parser.add_argument("-label_lung", type=int, help="Specify lung mask label to grow into (default: all labels considered part of that lung)")
    parser.add_argument("-erode", type=int, help="Optional lung mask erosion by before dilation (default: radius 3)")
    args = parser.parse_args()

    mask_vessel = sitk.ReadImage(args.mask_vessel) # Read labelled vessel mask
    seeds = morphological_closing_on_multilabel_mask(mask_vessel,30) # Close vessel mask to form compact seeds for dilation

    mask_lung = sitk.ReadImage(args.mask_lung) # Read lung mask
    if args.label_lung:
        mask_lung = sitk.BinaryThreshold(mask_lung, lowerThreshold=args.label_lung, upperThreshold=args.label_lung, insideValue=1, outsideValue=0) # Binarise lung mask for specified label
    else:
        mask_lung = sitk.BinaryThreshold(mask_lung, lowerThreshold=1, upperThreshold=2, insideValue=1, outsideValue=0) # Binarise lung mask for all nonzero labels

    if args.erode:
        mask_lung = binary_erosion(mask_lung,radius=args.erode) # Erode peripheral lung

    mask_lung = sitk.GetArrayFromImage(mask_lung) # Convert to numpy for processing
    mask_lung = np.where(mask_lung==4,0,mask_lung)
    mask_lobe = space_partitioning(mask_lung,sitk.GetArrayFromImage(seeds)) # For each labelled voxel, find closest lobe seed by Euclidean distance

    # Export lobe mask image
    mask_lobe = sitk.GetImageFromArray(mask_lobe)
    mask_lobe.CopyInformation(mask_vessel)

    if args.o:
        try:
            os.makedirs(os.path.dirname(args.o), exist_ok=True)
        except:
            pass # pass if output to current directory
        sitk.WriteImage(mask_lobe,args.o)
        print(f"Saved lobes mask to {args.o}")

        # steps={"1. Labelled vessel mask":"Output of get_vessel_mask.py is manually labelled in ITKSnap.",
        #         "2. Get seeds for space partitioning":"Obtained by morphological closing of labelled vessels",
        #         "3. Resultant lobe mask":"Based on method by Khiati 2024, HAL Open Science."
        #         }

        # np.savez_compressed('visualise_pipeline/vessel_to_lobe.npz',
        #                     steps=steps,
        #                     step1=sitk.GetArrayFromImage(mask_vessel),
        #                     step2=sitk.GetArrayFromImage(seeds),
        #                     step3=sitk.GetArrayFromImage(mask_lobe),
        #                     spacing=mask_lobe.GetSpacing())

if __name__ == "__main__":
    main()
