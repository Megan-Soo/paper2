#!/usr/bin/env python3
import SimpleITK as sitk
import numpy as np
import os, argparse

def left_hand_system(coords):
    """Check if the coordinates follow a right-hand rule system. x axis = Index finger. y axis = Middle finger. z axis = Thumb """
    
    coords = np.array(coords)

    x = coords[0,:]
    y = coords[1,:]
    z = coords[2,:]
    
    # Compute the cross product of X and Y
    z_calc = np.cross(x, y)
    return np.allclose(z, z_calc)

def convert_left_right_system(coords):
    """Convert right-hand coordinate system to left-hand system or vice versa."""
    coords = np.array(coords)

    coords[:, 2] = -coords[:, 2]  # Negate the Z axis to switch between left and right hand systems
    return coords

def estimate_densities(arr_masked,arr_mask,seq,TR,TE):
    """
    Estimate lung tissue densities from median-filtered MRI intensities normalised to muscle values.
    Parameters:
    - arr_masked: 3D numpy array of median-filtered MRI intensities within lung and muscle regions.
    - arr_mask: 3D numpy array of the lung and muscle mask (muscle label = 4).
    - seq: MRI sequence type ('UTE' or 'ZTE').
    - TR: Repetition time in ms.
    - TE: Echo time in ms. UTE: 0.06 ms; ZTE: 0.016 ms 3T Matai
    """
    if seq=='ZTE':
        TR = TR/1000/256 # (s) ZTE: divide by n spokes e.g. 629.2/1000/256 spokes 1.41 ms 3T Matai
    else:
        TR = TR/1000 # (s) UTE: just TR (s)
    t1_lung = np.exp(-TR / 1.4) # lung T1 1.4 s at 3T Nichols et al 2007
    t2star_lung = np.exp(-TE/0.74); # lung T2* 0.74 ms at 3T Yu et al 2011
    t1_muscle = np.exp(-TR / 1.1) # muscle T1 1.1 s at 3T Gold et al 2004
    t2star_muscle = np.exp(-TE / 25) # muscle T2* 25 ms at 3T Zaeske et al 2022
    lung_correction = t2star_lung / t1_lung
    muscle_correction = t2star_muscle / t1_muscle

    arr_muscle = np.where(arr_mask==4,arr_masked,0) # collect muscle voxels
    arr_muscle = arr_muscle[arr_muscle != 0]  # collect all non-zero values in a 1d array
    muscle_signal_mean = np.mean(arr_muscle)
    muscle_signal_corrected = muscle_signal_mean/muscle_correction

    # arr_masked = arr_masked[arr_masked != 0] # collect all non-zero values in a 1d array
    arr_lung_corrected = arr_masked/muscle_signal_corrected
    arr_lung_corrected = arr_lung_corrected/lung_correction
    return arr_lung_corrected, muscle_signal_corrected, lung_correction

def show_plot(arr_lung_corrected, arr_raw, subject, show=True, save=False, output_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def crop_image(image: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
        """Crops a 2D image array by removing specified pixels from each side.
        
        Args:
            image (np.ndarray): 2D array representing the image.
            top (int): Number of pixels to remove from the top.
            bottom (int): Number of pixels to remove from the bottom.
            left (int): Number of pixels to remove from the left.
            right (int): Number of pixels to remove from the right.
        
        Returns:
            np.ndarray: Cropped image.
        """
        return image[top:-bottom or None, left:-right or None]

    arr_combined = np.where(arr_lung_corrected>0,arr_lung_corrected,np.where(arr_raw>0,arr_raw,0))
    highlight = np.full_like(arr_combined,np.nan) # create empty arr
    highlight[arr_combined<1] = arr_combined[arr_combined<1]

    idx = arr_combined.shape[0]//2
    slice_all = arr_combined[idx,:,:]
    slice_higlight = highlight[idx,:,:]
    slice_all = crop_image(slice_all, top=50,bottom=50,left=20,right=20)
    slice_higlight = crop_image(slice_higlight,top=50,bottom=50,left=20,right=20)

    fig = plt.figure(figsize=(10, 4))  # Adjust figure size
    fig.canvas.manager.set_window_title(f'Subject {subject}')  # Set the figure window title
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1], width_ratios=[1, 1.1], wspace=0.20)  # add extra row below for alignment, 2nd col width is 1.1x width of 1st col, Minimize space between plots

    ax1 = plt.subplot(gs[0,0])

    ax1.axis("off")
    _ = ax1.imshow(slice_all,cmap='gray')
    im_highlight = ax1.imshow(slice_higlight,cmap='jet',vmin=0.0,vmax=1.0)

    cbar_ax = plt.subplot(gs[1,0]) # plot cbar in row below
    cbar = plt.colorbar(im_highlight,cax=cbar_ax,orientation='horizontal')
    cbar.set_label("Normalised density (g/cm3)",labelpad=5)
    cbar.ax.xaxis.set_label_position("bottom")

    # Histogram on left y-axis
    ax2 = plt.subplot(gs[0,1])
    data = arr_lung_corrected[arr_lung_corrected != 0]  # collect all non-zero values in a 1d array
    _, _, _ = ax2.hist(data, bins=30, weights=(np.ones_like(data) / len(data)) * 100, 
                                        alpha=0.6, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Normalised Density (g/cm$^3$)',fontsize=14)
    ax2.set_ylabel('Frequency (%)', color='steelblue', fontsize=14)
    ax2.set_xlim(0, 1)
    # ax2.set_ylim(0, 20)
    ax2.set_yticks(np.arange(0, 11, 2))  # Example: 0% to 20% in steps of 5
    ax2.tick_params(axis='y', labelcolor='steelblue')

    # Cumulative distribution on right y-axis
    ax3 = ax2.twinx()
    sorted_data = np.sort(data)
    cumulative = np.arange(1, len(sorted_data)+1) / len(sorted_data) * 100 # make it a percentage of the total
    ax3.plot(sorted_data, cumulative, color='darkorange', linewidth=2)
    ax3.set_ylabel('Cumulative Frequency (%)', color='darkorange', fontsize=14)
    ax3.tick_params(axis='y', labelcolor='darkorange')

    plt.title('Density Distribution')
    plt.grid(True)
    # plt.tight_layout()

    if save:
        if output_path is not None:
            plt.savefig(output_path,bbox_inches='tight', dpi=300)
            # print(f"Plot saved to {output_path}")
        else:
            print("Output path not provided. Plot not saved.")

    if show:
        plt.show(block=False)
        plt.pause(10)  # Keep it open for 2 seconds
        plt.close()

def get_bin_stats(arr_lung_corrected, num_bins=4):
    def split_bins(data,num_bins=4):
        """
        Sort the values then split them into bins with same number of values.
        """

        # Sort the data
        sorted_data = np.sort(data)

        # Calculate the number of values in each bin
        bin_size = len(sorted_data) // num_bins

        # Create bins
        bins = [sorted_data[i * bin_size:(i + 1) * bin_size] for i in range(num_bins)]

        # Get the edges of the bins
        bin_edges = [bin[0] for bin in bins] + [bins[-1][-1]]

        # Compute the histogram for the image data
        hist, _ = np.histogram(data, bins=bin_edges)

        return hist, bin_edges

    print(f"{np.count_nonzero(arr_lung_corrected)} voxels")

    # get lung density values (no muscle density values)
    densities = arr_lung_corrected[arr_lung_corrected != 0] # flatten to 1D array

    # split values into histogram with even bin areas
    hist, bin_edges = split_bins(densities,num_bins=num_bins)
    # Print the count of each bin
    for i in range(len(hist)):
        print(f"Bin {i + 1} ({bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}): {hist[i]} voxels")

    # get median value of each bin
    bin_medians = []
    for i in range(len(bin_edges)-1):
        bin_median = np.median(densities[(densities >= bin_edges[i]) & (densities < bin_edges[i+1])])
        bin_medians.append(bin_median)
    bin_medians = np.array(bin_medians)
    print(f"Bin medians: {[f'{median:.2f}' for median in bin_medians]}")

    return bin_edges, bin_medians

def label_bins(image_array, bin_edges):
    """
    Assigns labels to each voxel based on histogram bin edges.

    Parameters:
    - image_array: 3D numpy array representing the image.
    - bin_edges: 1D numpy array representing the edges of the histogram bins.

    Returns:
    - labeled_image: 3D numpy array where each voxel is assigned a bin label.
    """

    # Flatten image array and assign labels using np.digitize
    labeled_arr = np.digitize(image_array, bins=bin_edges, right=False)

    # Map values outside the valid bin range to zero
    labeled_arr[(image_array < bin_edges[0]) | (image_array >= bin_edges[-1])] = 0

    unique_labels, counts = np.unique(labeled_arr, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label != 0:
            print(f"Label {label}: {count} voxels")

    return labeled_arr

def export_arr_to_img(arr, output_path):
    image = sitk.GetImageFromArray(arr)
    image = sitk.Cast(image,sitk.sitkUInt8)
    image.CopyInformation(image)
    sitk.WriteImage(image, output_path)
    print(f" Saved bin-labelled mask to {output_path}")

def seed_lungs(bin_vals, arr_bin_labels, spacing):
    """
    Divide 32,000 units by the ratio of median bin densities in bin_vals.
    Uniformly sample units in each bin region given the labelled array, arr_bin_labels.
    Use spacing to generate 3D coordinates of the units.
    Estimate initial relative tissue unit volumes based on ratio of median densities in bin_vals.
    Scale relative tissue unit volumes such that they sum to target total volume, vol_total.
    """

    def split_value(total, values):
        """
        Splits a total amount into proportions based on a list of values.
        
        Parameters:
            total (float): The total amount to be split.
            values (list of float): A list of values representing proportions.
            
        Returns:
            list of int: A list containing the split values as integers.
        """
        # values = values**0.5 # power<1 to distribute more evenly across bins. values**0 gives uniform distribution. Power >1 allocates more units to last bin (high density bin)
        sum_values = sum(values)
        if sum_values == 0:
            return [0] * len(values)  # Avoid division by zero

        # Calculate the split values and round them to integers
        split_values = [(v / sum_values) * total for v in values]
        int_values = [int(round(v)) for v in split_values]

        # Adjust the last element to ensure the sum is correct
        int_values[-1] += total - sum(int_values)

        return int_values

    def generate_uniform_points_target(label_map, target_num_points, roi_label=1):
        """Generates a target number of uniform points inside a labeled ROI.
        
        If the ROI has fewer available voxels than the target, points are 
        duplicated with small random jittering (not ideal though).
        """
        indices = np.argwhere(label_map == roi_label)  # Get voxel coordinates within the ROI
        
        if len(indices) == 0:
            raise ValueError("The ROI is empty, no points available to sample.")

        if len(indices) >= target_num_points:
            # If we have enough points, randomly sample without replacement
            np.random.seed(39) # for reproducibility if we want to rerun with same mask
            selected_indices = np.random.choice(len(indices), target_num_points, replace=False)
            indices = indices[selected_indices]
            
        else:
            print(f"Warning: Not enough points in the ROI label {roi_label}. Oversampling with jittering.")
            # If we don't have enough points, oversample with small jittering
            num_repeat = target_num_points // len(indices)  # Whole repetitions
            remainder = target_num_points % len(indices)   # Extra points needed
            
            # Repeat points
            repeated_points = np.repeat(indices, num_repeat, axis=0)
            extra_points = indices[np.random.choice(len(indices), remainder, replace=False)]
            
            # Add small jitter (fraction of a voxel)
            jitter = (np.random.rand(len(repeated_points) + len(extra_points), 3) - 0.5) * 0.5
            
            # Combine indices
            indices = np.vstack((repeated_points, extra_points)) + jitter

        return np.array(indices)

    def convert_indices_to_coords(indices, spacing): # assumes isotropic spacing

        # Convert indices to real-world coordinates
        # indices[:,[0,2]] = indices[:,[2,0]] # this permanently reorders indices from ZYX to XYZ!
        coords = indices[:,::-1] * np.array(spacing) # preserve indices as ZYX but get coords as XYZ

        if not left_hand_system(coords): # pass x, y, z columns
            coords = convert_left_right_system(coords)
        
        return coords

    def inverse_densities_bins(bin_medians):
        return 1/bin_medians

    proportions = split_value(32000, bin_vals) # split 32k units based on ratio of median densities
    print(f"Seed proportions: {proportions}")

    # Generate seeds in lung
    indices = np.empty((0, 3))
    for i in range(len(proportions)):
        indices = np.vstack((indices,generate_uniform_points_target(arr_bin_labels, proportions[i], roi_label=i+1)))

    # Convert selected indices to coordinates
    coords = convert_indices_to_coords(indices, spacing).tolist() # zyx to xyz is done in convert_indices_to_coords().

    # Estimate initial volumes in lung
    vol_list = inverse_densities_bins(bin_vals) # Option 1: use bin medians

    print(f"Volume medians: {vol_list}")
    vols = []
    for vol,proportion in zip(vol_list,proportions):
        vols = vols + (proportion*[vol])
    # vols = inverse_densities_points(indices, arr_raw, muscle_signal_corrected, lung_correction).tolist() # Option 2: use voxel's approx density

    return indices.astype(int), np.array(coords), np.array(vols) # indices in z,y,x order, coords in x,y,z order

def extract_labels(label_array, indices_zyx):
    """
    Extract label values from a 3D labelled array at given voxel indices.

    Parameters
    ----------
    label_array : (Z, Y, X) ndarray
        The labelled 3D volume.
    indices_zyx : (N, 3) ndarray of int
        List of voxel indices in (z, y, x) order.

    Returns
    -------
    labels : (N,) ndarray
        Label values at the requested indices.
    """
    indices_zyx = np.asarray(indices_zyx)
    z, y, x = indices_zyx[:,0], indices_zyx[:,1], indices_zyx[:,2]
    return label_array[z, y, x]

def scale_vols(vals_unscaled,val_total):
    val_unscaled_total = sum(vals_unscaled)
    scale_factor = val_total/val_unscaled_total
    vols_scaled = [v*scale_factor for v in vals_unscaled] # Adjust volumes (mm3)
    return np.array(vols_scaled)

def dict2exnodedata(path, filename, my_dict, ext='.exnode'):
        file_path = os.path.join(path,filename)

        filename, extn = os.path.splitext(file_path)
        if bool(extn) is False:
            extn = ext

        # Check if all lengths are the same
        def check_lengths_consistent(my_dict):
            lengths = []
            
            for _, value in my_dict.items():
                length = len(value)
                lengths.append(length)

            # Check if all lengths are the same
            if len(set(lengths)) == 1:
                return lengths[0]  # All lengths are the same. Return one of them.
            else:
                return False  # Lengths are different

        # Check if all values have the same length
        length = check_lengths_consistent(my_dict)
        if length is False:
            print(" Number of field values don't match. Skipping export to exnode.")
            return
        
        num_fields = len(my_dict.keys())

        # Check if dict has specific node numbering
        for key, _ in my_dict.items():
            if 'Node number' in key:
                node_number = True # flag true
                node_numbers = my_dict[key] # extract the node numbers
                num_fields = num_fields-1
            else:
                node_number = False


        with open((filename + extn), 'w') as file:
            file.write(" Group name: \n")
            # file.write(" !#nodeset nodes\n")
            file.write(f' #Fields={num_fields}\n')

            # Write the header with order
            value_index = 1 # initialise value index
            for index, key in enumerate(my_dict.keys(), 1):  # Start index from 1
                if 'Node number' in key: # do not write Node numbers as field values
                    continue

                if key[0] == 'coordinates':  # 2D array (coordinates)
                    file.write(f" {index}) coordinates, coordinate, rectangular cartesian, #Components={key[1]}\n")
                    file.write(f"  x.  Value index= {value_index}, #Derivatives=0, #Versions=1\n")
                    value_index += 1
                    file.write(f"  y.  Value index= {value_index}, #Derivatives=0, #Versions=1\n")
                    value_index += 1
                    file.write(f"  z.  Value index= {value_index}, #Derivatives=0, #Versions=1\n")
                    value_index += 1
                elif key[1]>1: # if another field value with multiple components
                    file.write(f" {index}) {key[0]}, field, rectangular cartesian, #Components={key[1]}\n")
                    for count in range(key[1]):
                        file.write(f"  {count+1}.  Value index= {value_index}, #Derivatives=0, #Versions=1\n")
                        value_index += 1
                else:
                    file.write(f" {index}) {key[0]}, field, rectangular cartesian, #Components={key[1]}\n")
                    file.write(f"  1. Value index= {value_index}, #Derivatives=0\n")
                    value_index = value_index+1

            for i in range(length):
                if node_number:
                    file.write(f" Node:       {node_numbers[i]}\n")
                else:
                    file.write(f" Node:       {i+1}\n")  # Point N
                
                for key, value in my_dict.items():
                    if isinstance(value, np.ndarray): # convert np arrays here so no need to convert before entering function
                        value = value.tolist()

                    if 'Node number' in key: # do not write Node numbers as field values
                        continue

                    if isinstance(value[i], (list, np.ndarray)):  # If it's a list of coordinates
                        for coord in value[i]:
                            file.write(f"{str(round(coord,6)).rjust(15)}\n")  # Right-align the coordinate (adjust width as needed)
                    else:
                        file.write(f"{str(round(value[i],6)).rjust(15)}\n")  # Right-align the coordinate (adjust width as needed)

def array2ipdata(path,filename, array):
    file_path = os.path.join(path,filename)

    file_path, ext = os.path.splitext(file_path)
    if bool(ext) is False:
        ext = '.ipdata'
    idx=1
    with open(file_path+ext,'w') as f:
        f.write(f'{filename}\n')
        for x,y,z in array:
            f.write(f' {idx} {x}  {y}  {z}  1.0  1.0  1.0\n')
            idx=idx+1

def binary_erosion_selected(image, labels, radius=2, out=None):
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelRadius([radius, radius, radius])
    eroder.SetForegroundValue(1) # this only erodes pixels with value 1

    result = sitk.Image(image.GetSize(),sitk.sitkUInt8)
    result.CopyInformation(image)

    for lbl in labels:
        # 1) Extract label → binary mask
        mask = sitk.BinaryThreshold(image, lowerThreshold=lbl, upperThreshold=lbl,
                                    insideValue=1, outsideValue=0)

        # 2) Erode binary mask
        eroded = eroder.Execute(mask)

        # 3) Restore label value
        eroded_label = eroded * lbl

        # 4) Add back to result
        result = result + sitk.Cast(eroded_label, result.GetPixelID())

    if out is not None:
        sitk.WriteImage(result, out)

    return result

def apply_median_filter(image, radius=2, out=None):
    """
    Apply median filtering to a 3D SimpleITK image.

    Parameters:
    - image (sitk.Image): Input 3D volumetric image.
    - radius (int or tuple): Radius of the median filter kernel. Can be an integer or a 3D tuple.

    Returns:
    - sitk.Image: Filtered 3D image.
    """
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius(radius)
    filtered = median_filter.Execute(image)
    filtered.CopyInformation(image)
    if out is not None:
        sitk.WriteImage(filtered,out)
    return filtered

def main():
    parser = argparse.ArgumentParser(description="Create median-filtered image, erodes lobe mask, shifts init vol sum to lung vol instead of lobe vol." \
                                                "Estimate lung tissue density from median-filtered image (normalised to muscle values)." \
                                                "Use densities to get spatial distribution of tissue units across both lungs." \
                                                "Use lobe mask to group generated tissue units into respective lobes and get volume distribution per lobe.")
    parser.add_argument("-img", required=True, help="Path to raw image file (e.g. image.nii.gz)")
    parser.add_argument("-mask_lung_muscle", required=True, help="Path to (un-eroded) lung + muscle mask file (muscle must be label 4!")
    parser.add_argument("-mask_lobe", required=True, help="Path to (un-eroded) lobe mask file WITH VESSELS (shouldn't have label 4)")
    parser.add_argument("-mask_vessels", required=True, help="Path to vessel mask file (shouldn't have label 4). Remove vessels AFTER eroding each lobe.")
    parser.add_argument("-outdir", required=True, help="DIR to save generated tissue units")
    parser.add_argument("-seq", type=str, default='UTE', choices=['UTE','ZTE'], help="MRI sequence type: UTE or ZTE (default: UTE)")
    parser.add_argument("-tr", type=float, default=2.928, help="Repetition time TR in ms (default: 2.928 ms for UTE 3T Matai)")
    parser.add_argument("-te", type=float, default=0.060, help="Echo time TE in ms (default: 0.06 ms for UTE 3T Matai)")
    parser.add_argument("-bins", type=int, default=10, help="No. of regions to split tissue densities (default: 10)")
    parser.add_argument("-erode", type=int, default=2, help="Radius to erode each lobe")
    parser.add_argument("-save",type=str,default='y',help='y/n to save files.')
    args = parser.parse_args()

    # === Parameters
    plot = False
    save = args.save
    n_bins = args.bins
    output_dir = args.outdir
    os.makedirs(output_dir, exist_ok=True)
    subject = os.path.basename(os.path.normpath(output_dir))

    # === Read files
    mask_lung_muscle = sitk.ReadImage(args.mask_lung_muscle) # Read lung muscle mask (muscle label == 4 !!)
    mask_lobe = sitk.ReadImage(args.mask_lobe)
    mask_vessels = sitk.ReadImage(args.mask_vessels)
    image = sitk.ReadImage(args.img) # Read raw image
    step2a = np.flip(np.transpose(sitk.GetArrayFromImage(sitk.Mask(image,mask_lung_muscle)),(2,1,0)),2)
    if save=='y':
        image = apply_median_filter(image, out=os.path.join(output_dir,'med_filtered.nii.gz')) # Apply median filter
    else:
        image = apply_median_filter(image)
    spacing = mask_lobe.GetSpacing()
    step2b = np.flip(np.transpose(sitk.GetArrayFromImage(sitk.Mask(image,mask_lung_muscle)),(2,1,0)),2)

    step1a = np.flip(np.transpose(sitk.GetArrayFromImage(mask_lobe),(2,1,0)),2)
    # Erode lobe-by-lobe
    mask_lobe = binary_erosion_selected(mask_lobe,[1,2,3,5,6],args.erode)
    step1b = np.flip(np.transpose(sitk.GetArrayFromImage(mask_lobe),(2,1,0)),2)

    # Remove vessels afer eroding lobes
    inv_vessels = 1-sitk.BinaryThreshold(mask_vessels,lowerThreshold=1,upperThreshold=1000) # invert mask
    mask_lobe = sitk.Mask(mask_lobe,inv_vessels,outsideValue=0) # mask out vessels. this binarises the lobbe labels tho.
    step1c = np.flip(np.transpose(sitk.GetArrayFromImage(mask_lobe),(2,1,0)),2)

    # === MRI-quantified tissue density
    arr_lung_muscle = np.where(sitk.GetArrayFromImage(mask_lung_muscle)==4,4,sitk.GetArrayFromImage(mask_lobe)) # consider densities in "core" physiological region
    arr_intensities = np.where(arr_lung_muscle>0,sitk.GetArrayFromImage(image),0) # Get lung & muscle intensities for density quantification from median-filtered image

    # Get density map using muscle values
    arr_densities, _, _ = estimate_densities(arr_intensities,arr_lung_muscle,args.seq,args.tr,args.te) # feed lung & muscle intensities arr, and any arr w/ muscle mask (label 4)
    img_densities = sitk.GetImageFromArray(arr_densities)
    img_densities.CopyInformation(image)
    if save=='y':
        output_path = os.path.join(output_dir,'density_mask.nii.gz') # Export densities image
        sitk.WriteImage(img_densities,output_path)
        print(f" Saved density mask to {output_path}")
    step3 = np.flip(np.transpose(arr_densities,(2,1,0)),2)

    # Estimate mean lung density
    arr_densities = np.where(arr_lung_muscle == 4, 0, arr_densities) # remove muscle density values
    mean_density = np.mean(arr_densities[arr_densities != 0]) # get mean density
    std_density = np.std(arr_densities[arr_densities != 0]) # get stdev density
    print(f" Mean lung density: {mean_density:.2f} +- {std_density:.2f} g/cm3")
    if save=='y':
        show_plot(arr_densities,sitk.GetArrayFromImage(image), subject,show=plot, save=True, output_path=os.path.join(output_dir,'density_histogram.png')) # Show histogram
    else:
        show_plot(arr_densities,sitk.GetArrayFromImage(image), subject,show=plot, save=False, output_path=os.path.join(output_dir,'density_histogram.png')) # Show histogram

    print(f" Number of bins: {n_bins}")
    bin_edges, bin_medians = get_bin_stats(arr_densities,num_bins=n_bins) # Split lung tissue density into n_bins regions

    # === Tissue unit generation
    arr_bin_label = label_bins(arr_densities, bin_edges) # Assign unique labels to bins
    if save=='y':
        export_arr_to_img(arr_bin_label,os.path.join(output_dir, 'density_bins.nii.gz')) # Export bin labels mask
    step4 = np.flip(np.transpose(arr_bin_label,(2,1,0)),2)

    # Sample tissue units per region based on median density
    indices_all, coords_all, vols_unscaled_all = seed_lungs(bin_medians, arr_bin_label, spacing) # Submodule to distribute target tissue units across bins then derive coordinates and relative volumes
    print(f" Total number of points in both lungs: {len(coords_all)}")
    step5_1a = np.asarray(coords_all)

    parenchyma = sitk.Mask(mask_lung_muscle, inv_vessels) # remove vessels
    arr_parenchyma = np.where(sitk.GetArrayFromImage(parenchyma)!=4,sitk.GetArrayFromImage(parenchyma),1) # remove muscle voxels
    frc_vol = np.count_nonzero(arr_parenchyma) * np.prod(spacing) # FRC lung volume from (un-eroded) lung mask w/o vessels
    vols_scaled_frc = scale_vols(vols_unscaled_all, val_total=frc_vol) # Scale unit vols to sum to FRC lung volume. Large init vol range, large deviation from MoColoR esp anterior to hilum.
    print(f" Total volume of generated tissue units (scaled to FRC): {sum(vols_scaled_frc)/10**6:.2f} L") # Check total volume of generated units against FRC lung volume
    step5_1b = np.asarray(vols_scaled_frc)

    assert len(coords_all) == len(vols_scaled_frc) == len(indices_all), "Length of coords, vols, and indices must be the same."
    if save=='y':
        np.savez_compressed(os.path.join(output_dir,'tissue_units.npz'), coords=coords_all, vols=vols_scaled_frc, indices=indices_all, spacing=spacing) # Export coords, vols, and indices in compressed npz

    import polyscope as ps
    ps.init()
    vol_lobes = 0
    for (lobe,label) in zip(['RUL','RML','RLL','LUL','LLL'],[1,2,3,5,6]):
        vals = extract_labels(sitk.GetArrayFromImage(mask_lobe),indices_all) # get array of labels corresponding to indices
        indices_lobe = indices_all[vals==label] # Get indices of tissue units in lobe
        coords_lobe = coords_all[vals==label] # get coords of tissue units in lobe
        vols_lobe = vols_scaled_frc[vals==label] # Get init vol of tissue units in lobe
        vol_lobe = np.sum(np.array(vols_lobe))

        pc = ps.register_point_cloud(f"{lobe}",np.array(coords_lobe))
        pc.add_scalar_quantity("init vol", np.array(vols_lobe), enabled=True,vminmax=[np.min(vols_scaled_frc),np.max(vols_scaled_frc)]) # Add init vol as scalar quantity to tissue units for visualisation. Will be mapped to terminal Node numbers after growing.

        print(f" {len(coords_lobe)} tissue units in {lobe} ({vol_lobe/10**3:.2f} mL) ")
        vol_lobes = vol_lobes + vol_lobe # Tally up lobe volumes (to check against Lung Volume later)

        if save=='y':
            # --- Export exdata ---
            dict_lobe = {('coordinates',3):coords_lobe, ('init vol',1):vols_lobe, ('indices ZYX',3):indices_lobe} # Store in dict for export
            dict2exnodedata(output_dir, lobe, dict_lobe, ext='.exdata') # Export coords & vols in exdata. Vols will be mapped to terminal Node numbers after growing.
            print(f" Saved generated tissue units coordinates and volumes in {os.path.join(output_dir,lobe) + '.exdata'}")
            
            # --- Export ipdata ---
            array2ipdata(output_dir, lobe, coords_lobe) # Export coords ipdata. For tree growing.
            print(f" Saved generated tissue units coordinates in {os.path.join(output_dir,lobe) + '.ipdata'}")

        # region: save progress
        if lobe=='RUL':
            step6_1a=coords_lobe
            step6_1b=vols_lobe
        elif lobe=='RML':
            step6_2a=coords_lobe
            step6_2b=vols_lobe
        elif lobe=='RLL':
            step6_3a=coords_lobe
            step6_3b=vols_lobe
        elif lobe=='LUL':
            step6_4a=coords_lobe
            step6_4b=vols_lobe
        elif lobe=='LLL':
            step6_5a=coords_lobe
            step6_5b=vols_lobe
        # endregion

    print(f" Total volume of lobes: {vol_lobes/10**6:.2f} L") # Sanity check against Lung Volume derived earlier (should be identical)
    print(f"Init vol range {min(vols_scaled_frc):.2f} mm3 to {max(vols_scaled_frc):.2f} mm3")

    # units = ps.register_point_cloud("units", coords_all,enabled=True)
    # units.add_scalar_quantity("init vol", vols_scaled_frc, enabled=True) # Add init vol as scalar quantity to tissue units for visualisation. Will be mapped to terminal Node numbers after growing.
    
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("none")
    ps.set_navigation_style("free")
    ps.set_background_color([0, 0, 0])
    # ps.show()

    steps = {"1a. Labelled lobes mask": "Output of vessel_to_lobe.py",
             f"1b. Erode lobes mask by kernel radius {args.erode}": f"Leaving behind the 'core' region where conducting airways are likely to occupy.",
             "1c. Remove vessels from lobes mask":"Retain lung parenchyma voxels for calculating normalised lung tissue density.",
             "2a. Raw lung MRI": "",
             "2b. Median-filtered lung MRI": "Apply a median filter with radius 2. [med_filtered.nii.gz].",
             "3. Normalised tissue density":"Median-filtered lung parenchyma signals normalised against the average chest wall muscle signal \
                 & corrected for lung & muscle decay times at 3T. [density_mask.nii.gz, density_histogram.png].",
             "4. Density bins":"Divide the density range into 10 bins and assign labels to the respective regions. [density_bins.nii.gz]",
             "5. Generate acini tissue units":"Median density values are used to obtain the relative spatial distribution of acini and their relative volumes.\
                 Linearly scale the acini volumes so that their total sums to the volume of lung parenchyma voxels. [tissue_units.npz]",
             "6. Assign acini to lobes":"For each lobe, export [.ipdata] for grow_lobes.py & save acini volumes in [.exdata]."
            }

    np.savez_compressed('visualise_pipeline/generate_tissue_units_001.npz',
                        steps=steps,
                        step1a=step1a,
                        step1b=step1b,
                        step1c=step1c,
                        step2a=step2a,
                        step2b=step2b,
                        step3=step3,
                        step4=step4,
                        step5_1a=step5_1a,
                        step5_1b=step5_1b,
                        step6_1a=step6_1a,
                        step6_1b=step6_1b,
                        step6_2a=step6_2a,
                        step6_2b=step6_2b,
                        step6_3a=step6_3a,
                        step6_3b=step6_3b,
                        step6_4a=step6_4a,
                        step6_4b=step6_4b,
                        step6_5a=step6_5a,
                        step6_5b=step6_5b,
                        spacing=spacing
                        )

if __name__ == "__main__":
    main()
