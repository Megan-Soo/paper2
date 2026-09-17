import pyvista as pv
import meshlib.mrmeshpy as mr
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, generate_binary_structure, median_filter, center_of_mass
from scipy.spatial.distance import cdist
import skimage.measure
import stl
from stl import mesh
import os, re
from scipy.io import loadmat

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

def remove_small_components(mask, min_size=100):
    # Connected Component Analysis: Label connected regions and remove small components based on size.
    s = generate_binary_structure(2,2) # Generate a structuring element that will consider features connected even if they touch diagonally
    cleaned_mask = np.zeros_like(mask)
    for label_val in np.unique(mask):
        if label_val == 0:
            continue
        # Extract binary mask for current label
        binary_mask = mask == label_val
        # Label connected components
        labeled_array, num_features = label(binary_mask, structure=s)
        for region in range(1, num_features + 1):
            # Keep regions larger than min_size
            if np.sum(labeled_array == region) >= min_size:
                cleaned_mask[labeled_array == region] = label_val
    return cleaned_mask

def get_largest_components_sitk(multilabel_array): # multithreaded -> faster than scipy's single threaded label process (have issues)
    cleaned_mask = np.zeros_like(multilabel_array)

    for label_val in np.unique(multilabel_array):
        if label_val == 0:
            continue
        # Extract binary mask for the current label
        binary_mask = (multilabel_array == label_val).astype(np.uint8)

        # Convert to SimpleITK image
        binary_image = sitk.GetImageFromArray(binary_mask)

        # Connected Component Analysis
        labeled_image = sitk.ConnectedComponent(binary_image)

        # Relabel components by size (largest first)
        relabeled_image = sitk.RelabelComponent(labeled_image, sortByObjectSize=True)

        # Extract the largest component
        largest_component = sitk.GetArrayFromImage(relabeled_image) == 1

        # Add the largest component back to the cleaned mask
        cleaned_mask[largest_component] = label_val

    return cleaned_mask

def map_label_to_compartment(mask):
    if mask.ndim ==2:
        pass
    elif mask.ndim==3:
        # current array is in z,x,y format. axis = 0 (sup-inf), axis = 1 (med-lat), axis = 2 (ant-post).
        # we want to process in coronal view ie axis = 0 (med-lat), axis = 1 (sup-inf), axis = 2 (ant-post)
        mask = np.transpose(mask, (2, 1, 0)) # Rotate to x, y, z format

    # Step 1: Find centroids for each label
    labels = np.unique(mask)
    labels = labels[labels != 0]  # Exclude background (label 0)
    centroids = {label: np.array(center_of_mass(mask == label)) for label in labels}
    
    # Step 2: Split labels into left and right based on middle x value
    all_x = np.array([c[0] for c in centroids.values()])
    # Compute the midpoint of the x-range
    min_x = min(all_x)
    max_x = max(all_x)
    midpoint_x = (min_x + max_x) / 2
    left_labels = [label for label, centroid in centroids.items() if centroid[0] > midpoint_x]
    right_labels = [label for label, centroid in centroids.items() if centroid[0] <= midpoint_x]

    if len(left_labels)==1 and len(right_labels)==1: # if not lobes, just lungs
        label_mapping = {
                        right_labels[0]: "right",
                        left_labels[0]: "left",
                        }
        return label_mapping
    
    # Ensure correctness
    try:
        assert len(left_labels) == 2 and len(right_labels) == 3, f" Midpoint x: {midpoint_x}\n Left labels: {left_labels}\n Right labels: {right_labels}"
    except:
        # Combine all labels
        all_labels = left_labels + right_labels
        
        # Redistribute the labels
        left_labels = all_labels[:2]  # Assign the first 2 labels to left
        right_labels = all_labels[2:]  # Assign the remaining labels to right
    
    # Step 3: Find 'rightmiddle' centroid
    right_centroids = np.array([centroids[label] for label in right_labels])
    rightmiddle_label = right_labels[np.argmin(right_centroids[:, 1])]
    
    # Step 4: Find 'rightupper' centroid
    distances = cdist(right_centroids, right_centroids)
    distances[np.arange(len(distances)), np.arange(len(distances))] = np.inf  # Ignore self-distances
    farthest_index = np.argmax(distances.sum(axis=1))
    rightupper_label = right_labels[farthest_index]
    
    # Step 5: Find 'rightlower' centroid
    remaining_right_labels = set(right_labels) - {rightmiddle_label, rightupper_label}
    rightlower_label = remaining_right_labels.pop()
    
    # Step 6: Find 'leftupper' centroid (closest to 'rightupper' by z value)
    rightupper_centroid = centroids[rightupper_label]
    left_centroids = np.array([centroids[label] for label in left_labels])
    closest_to_rightupper = np.argmin(np.abs(left_centroids[:, 2] - rightupper_centroid[2]))
    leftupper_label = left_labels[closest_to_rightupper]
    
    # Step 7: Find 'leftlower' centroid
    remaining_left_labels = set(left_labels) - {leftupper_label}
    leftlower_label = remaining_left_labels.pop()
    
    # Map labels to string descriptions
    label_mapping = {
        rightmiddle_label: "RML",
        rightupper_label: "RUL",
        rightlower_label: "RLL",
        leftupper_label: "LUL",
        leftlower_label: "LLL",
    }
    
    return label_mapping

def smooth_mask_surface(mask, size=3):
    # Median Filtering: Smooth the boundaries by applying a median filter; reduces noise while preserving edges.
    smoothed_mask = np.zeros_like(mask)
    for label_val in np.unique(mask):
        if label_val == 0:
            continue
        binary_mask = mask == label_val
        smoothed_binary = median_filter(binary_mask.astype(np.uint8), size=size)
        smoothed_mask[smoothed_binary > 0] = label_val
    return smoothed_mask

def smooth_hilum(multilabel_mask, export_path,r=25):
    # Perform binary morphological closing
    # Closing = Dilation followed by Erosion
    radius = [r, r, r]  # Dilation radius for all axes

    # Convert to numpy array for easier processing
    multilabel_array = sitk.GetArrayFromImage(multilabel_mask)

    # Create an empty array for the resulting multilabel mask
    result_array = np.zeros_like(multilabel_array, dtype=np.uint8)

    # Get all unique labels in the mask (excluding background label 0 if desired)
    unique_labels = np.unique(multilabel_array)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background (optional)

    # Generate binary masks for each label
    for label in unique_labels:
        # Create a binary mask for the current label
        binary_mask_array = (multilabel_array == label).astype(np.uint8)
        
        # Convert back to SimpleITK image
        binary_mask = sitk.GetImageFromArray(binary_mask_array)
        binary_mask.CopyInformation(multilabel_mask)  # Preserve metadata

        closed_mask = sitk.BinaryMorphologicalClosing(binary_mask, radius)

        # Convert the closed mask back to numpy array
        closed_mask_array = sitk.GetArrayFromImage(closed_mask)

        # Add the closed binary mask back to the multilabel array
        result_array[closed_mask_array > 0] = label

    result_array = get_largest_components_sitk(result_array)

    # Convert the result array back to a SimpleITK image
    result_multilabel_mask = sitk.GetImageFromArray(result_array)

    # Copy metadata from the original multilabel mask
    result_multilabel_mask.CopyInformation(multilabel_mask)

    # Save the resulting multilabel mask
    sitk.WriteImage(result_multilabel_mask, export_path)
    print(f' Smoothed hilum region. Exported {export_path}')

    return

def decimate_file_size(mesh_path, target_size):
    file_size = os.path.getsize(mesh_path)
    mesh = mr.loadMesh(mesh_path)
    
    if file_size < target_size:
        return
    else:
        print(' Decimating saved mesh...')
    
    while file_size > target_size:
        prev_mesh = mesh # save a copy of current mesh

        settings = mr.DecimateSettings()
        settings.maxError = 0.5 # decimate it with max possible deviation 0.5
        result = mr.decimateMesh(mesh, settings)

        if result.vertsDeleted == 0 and result.facesDeleted == 0: # can't decimate anymore
            mesh = prev_mesh # restore previous iteration
            mr.saveMesh(mesh,mesh_path) # save and exit
            return

    print(f' Final file size: {os.path.getsize(mesh_path)}')
    return

def convert_stl_to_ply(stl_path):
    mesh = pv.read(stl_path)
    path_no_ext = os.path.splitext(stl_path)[0]
    mesh.save(path_no_ext +'.ply', binary=False)

def image_to_stl(image, orientation, out_dir, suffix=False):

    image = sitk.DICOMOrient(image, orientation)
    spacing = list(image.GetSpacing())
    spacing = [spacing[2], spacing[1], spacing[0]] # skimage takes coordinates in zyx format. Swap 1st and 3rd columns to get zyx format.
    image_array = sitk.GetArrayFromImage(image)

    # # Connected Component Analysis (in [x,y,z] format): Label connected regions and remove small components based on size.
    # image_array = remove_small_components(image_array, min_size=1000)
    image_array = get_largest_components_sitk(image_array)

    # Map label values to lobes
    label_mapping = map_label_to_compartment(image_array) # `xyz_array` is your 3D multi-label mask
    # print(label_mapping)

    for label, lobe in label_mapping.items():
        print(f' Generating mesh for label {int(label)}, {lobe}')

        dataarray = (image_array == label)
        # Median Filtering: Smooth the boundaries
        dataarray = smooth_mask_surface(dataarray, size=3)
        
        # Generate mesh using marching cubes
        verts, faces, _, _ = skimage.measure.marching_cubes(dataarray, spacing=spacing)
        verts = verts[:, [2,1,0]] # skimage takes coordinates in zyx format. Swap 1st and 3rd columns - zyx -> xyz
        if not left_hand_system(verts): # pass x, y, z columns
            verts = convert_left_right_system(verts)
        meshobj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                meshobj.vectors[i][j] = verts[f[j],:]
        
        # Save the mesh
        if suffix is not False:
            full_path =  os.path.join(out_dir, lobe + '_' + suffix +'.stl') 
        else:
            full_path =  os.path.join(out_dir, lobe +'.stl') 
        meshobj.save(full_path, mode=stl.Mode.ASCII)

        decimate_file_size(mesh_path=full_path, target_size=1000000) # size in bytes (1 Mb = 10e6 bytes)
        convert_stl_to_ply(stl_path=full_path)
        # Convert the faces array to PyVista format
        # Add the number of vertices (3 for triangles) before each face
        # faces_pv = np.hstack([[3] + list(face) for face in faces])
        # pv_mesh = pv.PolyData(verts,faces_pv)
    
    return

def array_to_stl(dataarray,spacing,out):
        spacing = [spacing[2], spacing[1], spacing[0]] # skimage takes coordinates in zyx format. Swap 1st and 3rd columns to get zyx format.
        
        # Median Filtering: Smooth the boundaries
        dataarray = smooth_mask_surface(dataarray, size=3)
        
        # Generate mesh using marching cubes
        verts, faces, _, _ = skimage.measure.marching_cubes(dataarray, spacing=spacing)
        verts = verts[:, [2,1,0]] # skimage takes coordinates in zyx format. Swap 1st and 3rd columns - zyx -> xyz
        if not left_hand_system(verts): # pass x, y, z columns
            verts = convert_left_right_system(verts)
        meshobj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                meshobj.vectors[i][j] = verts[f[j],:]
        
        # Save the mesh
        meshobj.save(out, mode=stl.Mode.ASCII)

        decimate_file_size(mesh_path=out, target_size=1000000) # size in bytes (1 Mb = 10e6 bytes)
        convert_stl_to_ply(stl_path=out)

def array_to_mesh(dataarray, spacing, skip_smoothing=False): # specially for compress_maps_translate_centre.py

    if not skip_smoothing:
        # Median Filtering: Smooth the boundaries
        dataarray = smooth_mask_surface(dataarray, size=3)
    
    # Generate mesh using marching cubes
    verts, faces, _, _ = skimage.measure.marching_cubes(dataarray, spacing=spacing)
    verts = verts[:, [2,1,0]] # skimage processes coordinates in zyx format. Return to xyz format when done. Swap 1st and 3rd columns - zyx -> xyz
    if not left_hand_system(verts): # pass x, y, z columns
        verts = convert_left_right_system(verts)
    meshobj = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            meshobj.vectors[i][j] = verts[f[j],:]
    return meshobj

def separate_multilabel_image(image,export_dir):
    multilabel_array = sitk.GetArrayFromImage(image)
    multilabel_array = get_largest_components_sitk(multilabel_array)
    # Map label values to lobes
    label_mapping = map_label_to_compartment(multilabel_array)
    for label, lobe in label_mapping.items():
        # Extract binary mask for current label
        label_mask = np.where(multilabel_array == label, label, 0)
        print(f' Exporting image for label {int(label)}, {lobe} lobe')
        img_to_write = sitk.GetImageFromArray(label_mask)
        img_to_write.CopyInformation(image)
        sitk.WriteImage(img_to_write, os.path.join(export_dir,f'mask_{lobe}.mha'))

def attach_artery_to_grown_airway(upper_artery, grown_airway):
    # delete Nodes 1-14 of grown_airway
    del grown_airway[0:14]
    # attach remaining grown nodes to upper_artery
    full_artery = np.append(upper_artery, grown_airway, axis=0)
    return full_artery

def mat2pydict(filepath, target_var):
    # Load the .mat file
    mat_data = loadmat(filepath)

    # List all keys in the .mat file (excluding metadata keys)
    keys = [key for key in mat_data.keys() if not key.startswith('__')]

    # Find keys containing any of the target substrings
    target_keys = [key for key in keys if any(re.search(sub, key, re.IGNORECASE) for sub in target_var)]

    # Extract only the variables matching the target keys
    extracted_data = {key: mat_data[key] for key in target_keys}

    return extracted_data

def combine_fields(my_dict):
    # Normalize all values to 2D arrays
    normalized = [
        np.atleast_2d(value).T if np.ndim(value) == 1 else np.array(value)
        for value in my_dict.values()
    ]

    # Ensure all arrays have the same number of rows
    max_rows = max(arr.shape[0] for arr in normalized)
    aligned = [np.broadcast_to(arr, (max_rows, arr.shape[1])) if arr.shape[0] == 1 else arr for arr in normalized]

    # Combine along the second axis (columns)
    combined_array = np.hstack(aligned)
    return combined_array

def update_dict(my_dict, results_to_update):
    # Convert results_to_update to a NumPy array for easier slicing
    results_array = np.array(results_to_update)

    # Iterate through the dictionary and update it with corresponding columns
    for (field, num_columns), value in my_dict.items():
        # Extract the corresponding number of columns
        columns = results_array[:, :num_columns].tolist()
        
        # Update the dictionary
        my_dict[(field, num_columns)] = columns

        # Remove the extracted columns from the results_array
        results_array = np.delete(results_array, slice(0, num_columns), axis=1)

    return my_dict

def list2dict(list_keys,list_2d):
    # Initialize the dictionary to hold the 2D lists
    result = {key: [] for key in list_keys}

    for row in list_2d:
        i=0
        for key in list_keys:
            if 'coordinates' in key:
                result[key].append(row[i:i+3])
                i=i+3
            else:
                result[key].append(row[i])
                i=i+1

    return result

def get_two_largest_components(label_array): # created for 2d slice analysis but may work for 3d array?
    
    # Convert to SimpleITK image
    binary_image = sitk.GetImageFromArray(label_array)

    # Connected Component Analysis
    labeled_image = sitk.ConnectedComponent(binary_image) # fullyConnected includes diagonal connections
    # Relabel components by size (largest first)
    relabeled_image = sitk.RelabelComponent(labeled_image, sortByObjectSize=True)

    # Get the labels of the two largest components
    label_array = sitk.GetArrayFromImage(relabeled_image)
    unique_labels = np.unique(label_array)

    # Remove the background label (usually label 0)
    unique_labels = unique_labels[unique_labels != 0]

    # Get the two largest components (the first two in the sorted list)
    two_largest_labels = unique_labels[:2]

    # Create a binary mask for the two largest components
    binary_mask = np.zeros_like(label_array, dtype=np.uint8)
    
    size_threshold = 50 # min no. of pixels that a component should have in a 2d slice
    for label in two_largest_labels:
        binary_mask[label_array == label] = label #not sure why this works in hist_analysis when binary_mask[label_array == label] = 1
        if np.count_nonzero(binary_mask) < size_threshold:
            binary_mask[label_array == label] = 0 # if too few elements (small component prolly not lung), reset to zero
    
    return binary_mask

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

def bias_correction(input_image: sitk.Image, mask: sitk.Image = None, num_iterations: int = 50, out=None) -> sitk.Image:
    """
    Perform full-resolution N4 bias field correction on a 3D image.

    Parameters
    ----------
    input_image : sitk.Image
        The 3D input image (grayscale or scalar type).
    mask : sitk.Image, optional
        Binary mask defining the region of interest for bias correction.
        If None, a mask will be automatically generated using Otsu thresholding.
    num_iterations : int, optional
        Number of iterations for each fitting level (default: 50).
    out : str, optional
        Path to export bias corrected image.

    Returns
    -------
    sitk.Image
        The bias-corrected image.
    """
    
    if input_image.GetDimension() != 3:
        raise ValueError("Input image must be 3D.")

    # Convert to float type (required by N4)
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    
    # If no mask provided, generate one using Otsu threshold
    if mask is None:
        mask = sitk.OtsuThreshold(input_image, 0, 1, 200)
    if mask is not None:
        mask = sitk.BinaryThreshold(mask,lowerThreshold=0, upperThreshold=float('inf'), insideValue=1, outsideValue=0)

    # Perform N4 bias field correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([num_iterations] * 4)
    corrected_image = corrector.Execute(input_image, mask)
    corrected_image.CopyInformation(input_image)

    if out is not None:
        sitk.WriteImage(corrected_image,out,useCompression=True)

    return corrected_image
