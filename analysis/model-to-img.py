#!/usr/bin/env python3
import argparse, os, re
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
from scipy import ndimage

def kmeans_cluster(values, n_clusters=5):
    """
    Cluster 1D values into exactly n_clusters groups.
    No assumptions about normality.
    
    Parameters
    ----------
    values : (N,) array
        Scalar values associated with each 3D point.
    n_clusters : int
        Number of clusters to produce.

    Returns
    -------
    labels : (N,) array of ints in [0, n_clusters-1]
        Cluster assignment for each value.
    """

    values = np.asarray(values).reshape(-1, 1)

    # K-means on 1D values
    km = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=0
    )
    labels = km.fit_predict(values)

    # Sort clusters by centroid value (optional but usually desired)
    order = np.argsort(km.cluster_centers_.flatten())
    remap = np.zeros_like(order)
    remap[order] = np.arange(n_clusters)

    return remap[labels]

def hierarchical_kmeans_midpoint_3d_masked(X3d, background_mask):
    """
    X3d: (H, W, D) grayscale volume in [0,255]
    background_mask: boolean array, True = foreground, False = background
    Returns: labels_3d with shape (H, W, D)
    Background voxels are assigned label 0.
    """

    def midpoint_initial_means(k, low=0, high=255):
        width = (high - low) / k
        mids = low + (np.arange(k) + 0.5) * width
        return mids.reshape(-1, 1)
    
    H, W, D = X3d.shape

    # Extract only foreground voxels
    fg_indices = np.where(background_mask)
    X_fg = X3d[fg_indices].reshape(-1, 1).astype(np.float32)

    # -------------------------------
    # Level 1: 4 clusters (C2–C5)
    # -------------------------------
    init_centroids_lvl1 = midpoint_initial_means(4)
    km1 = KMeans(n_clusters=4, init=init_centroids_lvl1, n_init=1, random_state=0)
    labels_lvl1 = km1.fit_predict(X_fg)  # 0..3
    # Sort the clusters by intensity to assign C2–C5 in order of increasing intensity
    centroids = km1.cluster_centers_.flatten()  # shape (4,)
    order = np.argsort(centroids)  # e.g., [2, 0, 3, 1]
    mapping = {old: new for new, old in enumerate(order)}
    labels_lvl1_sorted = np.array([mapping[l] for l in labels_lvl1])
    labels_lvl1 = labels_lvl1_sorted + 2        # → 2..5

    # -------------------------------
    # Level 2: refine C2 → C21–C24
    # -------------------------------
    C2_mask = labels_lvl1 == 2
    X_C2 = X_fg[C2_mask]

    km2 = KMeans(n_clusters=4, n_init=10, random_state=0)
    labels_C2 = km2.fit_predict(X_C2)  # 0..3
    # Sort the clusters by intensity to assign C21–C24 in order of increasing intensity
    centroids = km2.cluster_centers_.flatten()  # shape (4,)
    order = np.argsort(centroids)  # e.g., [2, 0, 3, 1]
    mapping = {old: new for new, old in enumerate(order)}
    labels_C2 = np.array([mapping[l] for l in labels_C2]) # 0..3, sorted by intensity

    # -------------------------------
    # Merge C21–C23 → C1, C24 → new C2
    # -------------------------------
    final_fg_labels = labels_lvl1.copy()
    C2_indices = np.where(C2_mask)[0]

    for sub in [0, 1, 2]:  # C21, C22, C23
        final_fg_labels[C2_indices[labels_C2 == sub]] = 1

    final_fg_labels[C2_indices[labels_C2 == 3]] = 2  # C24

    # -------------------------------
    # Scatter back into 3D volume
    # -------------------------------
    labels_3d = np.zeros((H, W, D), dtype=np.int32)
    labels_3d[fg_indices] = final_fg_labels

    return labels_3d

def natural_key(s):
    """
    Split string into list of text and integer chunks for natural sorting.
    e.g. 'img10.nii.gz' -> ['img', 10, '.nii.gz']
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)
    ]

def read_model_results(dir,debug=False):
    """
    Read model ventilation results from exelem files in the given directory.
    Return dictionary with keys: 'flow_t', 'flow_t_trach'
    """
    def read_exelem(file_path):
        
        # Read entire file as text
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Regex patterns
        float_int_pattern = r"[-+]?\d*\.\d+E[+-]?\d+" # Regex pattern to match float or integer numbers

        values = []
        for i in range(len(lines)):
            if 'Element' in lines[i]:
                values.append(float(re.findall(float_int_pattern, lines[i+2])[0]))

        return np.array(values)

    def read_exnodedata(file_path,extn='.exnode'): # returns dict with key:value - (field name, num Components):[array of field values]
        try:
            file_path, ext = os.path.splitext(file_path)
            if bool(ext) is False:
                ext = extn

            with open((file_path+ext), 'r') as file:
                lines = file.readlines()

            results = {}  # Dictionary to store the results
            for line in lines:
                if ')' in line:
                    # Find the closing parenthesis
                    close_paren_index = line.find(')')
                    # Find the next comma after the closing parenthesis
                    next_comma_index = line.find(',', close_paren_index)
                    if next_comma_index != -1:
                        # Extract the substring between ')' and ','
                        words_between = line[close_paren_index + 1:next_comma_index].strip()
                    else:
                        words_between = line[close_paren_index + 1:].strip()  # If no comma, get till the end

                    match = re.search(r"Components=(\d+)", line)
                    if match:
                        components_value = int(match.group(1))
                        key = (words_between, components_value)
                        results[key] = None

            ## Collect and store index of each node
            indices = [i for i, s in enumerate(lines) if 'Node' in s]
            int_pattern = r"\b\d+\b" # Regex pattern to match integer numbers
            float_int_pattern = r'[+-]?\d+(?:\.\d+)?' # Regex pattern to match float or integer numbers

            list_node_num = []
            for idx in range(len(indices)):
                line = lines[indices[idx]]
                node_number = int(re.findall(int_pattern, line)[-1])
                list_node_num.append(node_number)

            prev_num_components = 0
            iterator = iter(results.items())
            for field in range(len(results)):
                entry = next(iterator)
                key = entry[0]
                components = entry[0][1]
                # print("Key (field name, components):", key) # field name, components
                # print("Components:", components)

                my_list = []
                for idx in range(len(indices)):
                    for i in range(components):
                        line = lines[indices[idx]+1+prev_num_components+i]
                        value = float(re.findall(float_int_pattern, line)[0])
                        my_list.append(value)

                if components>1:
                    # Reshape into a 2D list
                    my_list = [my_list[i:i+components] for i in range(0, len(my_list), components)]

                results[key] = my_list

                prev_num_components = prev_num_components + components

            results[("Node number",None)] = list_node_num
            return results

        except FileNotFoundError:
            print(f"File not found: {file_path}")
            exit()
        except Exception as e:
            print(f"An error occurred: {e}")
            exit()

    # --- Read trachea flow(t) values from exelem files ---
    exelem_files = sorted(
        [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(".exelem") and "terminal" in f],
        key=natural_key
    ) # exelem files containing trachea flow(t) values

    flow_t_trach = []
    for f in exelem_files:
        flow_t_trach.append(read_exelem(f))

    # --- Read terminal flow(t) values from exnode files ---
    exnode_files = sorted(
        [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(".exnode") and "terminal" in f],
        key=natural_key
    ) # exnode files containing terminal acini flow(t) values

    assert len(exnode_files) != 0, "No exnode files found in the specified directory."
    if debug:
        print(f"Read exnode files: {exnode_files}")

    flow_t = []
    flow_proportion = []
    vol_t=[]
    vol_proportion=[]
    for f in exnode_files:
        dict_exnode = read_exnodedata(f,extn='.exnode')
        flow = np.array(dict_exnode[('flow',1)])
        flow_t.append(flow)
        fv = flow/np.sum(flow)
        flow_proportion.append(fv)
        vol_t.append(np.array(dict_exnode[('volume',1)]))
        vol_proportion.append(np.array(dict_exnode[('max volume',1)])/np.sum(np.array(dict_exnode[('max volume',1)])))
        
    coords=dict_exnode[('coordinates',3)]

    return {'vol_proportion':vol_proportion,'vol_t':vol_t, 'flow_t_proportion':flow_proportion,'flow_t': np.array(flow_t), 'flow_t_trach': np.array(flow_t_trach), 'coords': np.array(coords)}

def voxelise_point_cloud(points, values, voxel_size, grid_origin, grid_shape):
    """
    Convert point cloud data to a 3D volume.
    
    Parameters
    ----------
    points : (N, 3) array
        Coordinates of the points in the point cloud.
    voxel_size : float
        Voxel spacing in the output volume.
    grid_origin : tuple of floats
        Origin of the output volume in world coordinates.
    grid_shape : tuple of ints
        Shape of the output volume.
    values : (N,) array
        Values associated with each point in the point cloud.

    Returns
    -------
    dict with keys:
        'voxel_mean': (nx, ny, nz) array of mean values in each voxel
        'voxel_std': (nx, ny, nz) array of standard deviation in each voxel
        'voxel_cv': (nx, ny, nz) array of coefficient of variation in each voxel
        'voxel_count': (nx, ny, nz) array of counts of points in each voxel
        'valid_mask': (nx, ny, nz) boolean array indicating which voxels have at least one point
    """

    if points[0][-1] < 0: # if point cloud in negative z axis,
        points[:, 2] = points[:, 2]+(256*1.328) # shift to positive z axis by Bbox length
    points = points[:, ::-1] # Convert XYZ to ZYX

    idx = ((points - grid_origin) / voxel_size).astype(int)
    valid = (
        (idx[:,0] >= 0) & (idx[:,0] < grid_shape[0]) &
        (idx[:,1] >= 0) & (idx[:,1] < grid_shape[1]) &
        (idx[:,2] >= 0) & (idx[:,2] < grid_shape[2])
    )
    idx = idx[valid]
    vals = values[valid]
    assert vals is not None and len(vals) > 0, "No valid points found within the specified grid."

    voxel_sum = np.zeros(grid_shape, dtype=float)
    voxel_sum_sq = np.zeros(grid_shape, dtype=float)
    voxel_count = np.zeros(grid_shape, dtype=int)

    np.add.at(voxel_sum, tuple(idx.T), vals)
    np.add.at(voxel_sum_sq, tuple(idx.T), vals**2)
    np.add.at(voxel_count, tuple(idx.T), 1)

    voxel_mean = np.divide(voxel_sum, voxel_count, where=voxel_count>0)
    voxel_var  = np.divide(voxel_sum_sq, voxel_count, where=voxel_count>0) - voxel_mean**2
    voxel_std  = np.sqrt(np.maximum(voxel_var, 0))
    voxel_cv   = np.divide(voxel_std, voxel_mean, where=voxel_mean!=0)

    return {'voxel_sum':voxel_sum,'voxel_mean': voxel_mean, 'voxel_std': voxel_std, 'voxel_cv': voxel_cv, 'voxel_count': voxel_count, 'valid_mask': voxel_count>0}

def dilate_np(arr,indices, vals, max_size=5):
    """
        A labelled array, arr.
        Indices match labels to the values in vals.
        Values are normalised, then the range is scaled by max_size.
        Max size controls extent of label dilation.
    """
    vals = (vals-np.min(vals))/(np.max(vals)-np.min(vals)) # normalise values 0-1
    vals = np.array(vals*max_size,dtype=int) # range acini label radius 1-5

    for lab,(id,v) in enumerate(zip(indices,vals)):
        z,y,x = id
        # arr[z-v:z+v,y-v:y+v,x-v:x+v] = lab # dilate full
        arr[z-v//2:z+v//2,y-v//2:y+v//2,x-v//2:x+v//2] = lab # dilate half
    return arr # dilate full and dilate half give similar val distribution dorsoventrally

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

def assign_val(arr,vals):
    for lab,val in enumerate(vals):
        arr[arr==lab] = val
    return arr

def main():

    # Read lung mask and erode it to avoid edge effects in clustering
    mask = sitk.ReadImage(mask_path)
    mask = sitk.BinaryThreshold(mask, 1,2,1,0)  # Convert to binary mask
    mask_arr = sitk.GetArrayFromImage(mask)  # Convert to array
    if erode==1:
        mask_eroded = sitk.BinaryErode(mask, [3,3,3], sitk.sitkBall)  # Erode by 3 voxels
        mask_arr = sitk.GetArrayFromImage(mask_eroded)

    dict_model = read_model_results(model_folder,debug=False) # --- Read model ventilation phase results ---
    print(f"Read model results...") 
    n_phases = len(dict_model['flow_t'])
    bg = -1 # Set background==-1 for new images

    # --- Create acini labels in array ---
    coords = dict_model['coords']
    if coords[0][2]<0:
        coords[:,2] = -coords[:,2] # remove negative sign
    indices = coords[:,::-1] / mask.GetSpacing()[0]
    indices = np.array(indices,dtype=int)
    arr_new = np.zeros(mask_arr.shape)
    for i,id in enumerate(indices):
        arr_new[tuple(id)] = i+1 # assign 32k labels

    if outdir:
        # --- Write Image ---
        new = sitk.GetImageFromArray(arr_new)
        new.CopyInformation(mask)
        new = sitk.Cast(new,sitk.sitkUInt8)
        sitk.WriteImage(new,os.path.join(outdir,'acini_label.nii.gz'))

    # --- Space partition acini labels into eroded lung mask ---
    arr_labelled = space_partitioning(mask_arr, arr_new) #arr_labelled)
    arr_labelled = np.asarray(arr_labelled,dtype=int)

    if outdir:
        # --- Write Image ---
        new = sitk.GetImageFromArray(arr_labelled)
        new.CopyInformation(mask)
        new = sitk.Cast(new,sitk.sitkUInt8)
        sitk.WriteImage(new,os.path.join(outdir,f'spacePartition_acini.nii.gz'))
    
    for phase in range(n_phases):
        print(f"\nProcessing phase {phase+2}...")

        # --- kmeans clustering of acini vols ---
        values = dict_model['vol_t'][phase] # assign flow values to corresponding labels
        cluster_model = kmeans_cluster(values,n_clusters=5) +1 # 1-indexing
        print(f"Kmeans vol VDP = {values[cluster_model==1].sum()/values.sum():.2%}")

        # --- kmeans clustering of acini flow ---
        values = np.abs(dict_model['flow_t'][phase]) # assign flow values to corresponding labels. flow is directional, remove -ve sign
        cluster_model = kmeans_cluster(values,n_clusters=5) +1 # 1-indexing
        values = dict_model['vol_t'][phase] # use volumes to calculate vent VDP bc volumes account for spatial defect
        print(f"Kmeans flow VDP = {values[cluster_model==1].sum()/values.sum():.2%}")
        
        # --- Assign values to labels ---
        values = cluster_model
        res = np.full(arr_labelled.shape, 0, dtype=values.dtype)
        res[mask_arr>0] = values[arr_labelled[mask_arr>0] - 1] # labels start from 1, but for indexing, start from 0

        if outdir:
            # --- Write Image ---
            new = sitk.GetImageFromArray(res)
            new.CopyInformation(mask)
            new = sitk.Cast(new,sitk.sitkInt8)
            sitk.WriteImage(new,os.path.join(outdir,f'kmeans_clusters{phase+2}.nii.gz'))
        
        # --- Assign values to labels ---
        values = dict_model['flow_t'][phase] # assign flow values to corresponding labels
        res = np.full(arr_labelled.shape, bg, dtype=values.dtype)
        res[mask_arr>0] = values[arr_labelled[mask_arr>0] - 1] # labels start from 1, but for indexing, start from 0

        if outdir:
            # --- Write Image ---
            new = sitk.GetImageFromArray(res)
            new.CopyInformation(mask)
            new = sitk.Cast(new,sitk.sitkFloat32)
            sitk.WriteImage(new,os.path.join(outdir,f'flow{phase+2}.nii.gz'))

        # --- Create Fractional Ventilation Image --- 
        res = np.where(res==bg,res, np.abs(res)) # flow is directional. Analyse magnitudes only, remove -ve signs.
        res = np.where(res == bg, res, (res - res[res != bg].min()) / # normalise ei to 0-255 range, ignoring background -1 values
                        (res[res != bg].max() - res[res != bg].min())*255)

        if outdir:
            # --- Write Image ---
            new = sitk.GetImageFromArray(res)
            new.CopyInformation(mask)
            new = sitk.Cast(new,sitk.sitkFloat32)
            sitk.WriteImage(new,os.path.join(outdir,f'fv{phase+2}.nii.gz'))
        
        # --- Hierarchical kmeans clustering ---
        cluster_model = hierarchical_kmeans_midpoint_3d_masked(res,mask_arr>0)
        print(f"H-Kmeans flow VDP = {np.sum(cluster_model==1)/np.sum(cluster_model>0):.2%}")
        
        if outdir:
            # --- Write Image --- 
            cluster_model = sitk.GetImageFromArray(cluster_model)
            cluster_model.CopyInformation(mask)
            sitk.WriteImage(cluster_model, os.path.join(outdir, f"hkm_clusters{phase+2}.nii.gz"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical kmeans clustering on model and MoCoLoR ventilation distribution and evaluate DICE score for each cluster.")
    parser.add_argument("model_folder", type=str, help="Path to subject folder containing model ventilation results.")
    parser.add_argument("mask_path", type=str, help="Path to lung mask NIfTI file.")
    parser.add_argument("-erode", type=int, default=1, help="1:True. Match physical space of masked Jac images. (Erosion sig affects n voxels & thus VDP %%)")
    parser.add_argument("-outdir", type=str, required=False, help="Directory for model 'images'.")
    args = parser.parse_args()
    model_folder = args.model_folder # 'results/EXAM5332_10phases/gradwarp/'
    mask_path = args.mask_path # '../MoCoLoR/EXAM5332_10phases/struct_gradwarp/masks/mocolor_nii_phase_1.nii.gz'
    erode = args.erode
    outdir = args.outdir #os.path.join(model_folder, "vol-to-img")
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    main()
