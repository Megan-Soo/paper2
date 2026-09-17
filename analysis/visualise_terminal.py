import polyscope as ps
import numpy as np
import re,os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def midpoint_initial_means(k, low=0, high=255):
    width = (high - low) / k
    mids = low + (np.arange(k) + 0.5) * width
    return mids.reshape(-1, 1)

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

def extract_coordinates(file_path):
    coordinates = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Node number"):
            i += 1
            x = float(lines[i].split(":")[-1].strip())
            i += 1
            y = float(lines[i].split(":")[-1].strip())
            i += 1
            z = float(lines[i].split(":")[-1].strip())
            coordinates.append([x, y, z])
        i += 1
    
    return np.array(coordinates)

def extract_global_numbers(file_path):
    global_numbers = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Element number"):
            while not lines[i].strip().startswith("Enter the 2 global numbers"):
                i += 1
            numbers = list(map(lambda x: int(x), lines[i].split(":")[-1].strip().split()))
            global_numbers.append(numbers)
        i += 1
    
    return np.array(global_numbers)

def extract_radius(file_path):
    radius_values = []
    pattern = re.compile(r'The field variable value is \[ .*?\]: ([\d\.D\+\-]+)')
    
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                value = match.group(1).replace('D', 'E')  # Convert Fortran-style exponent
                radius_values.append(float(value))
    
    return np.array(radius_values)

def compute_joint_radii(nodes, edges, edge_mid_radii):
    """
    Given a radius for each edge (at midpoint), compute per-node radius
    as the average of connected edge radii. If a node has only one incident
    edge, use that edge's radius directly.
    
    Parameters
    ----------
    nodes : (N,3)
    edges : (E,2)
    edge_mid_radii : (E,)
    
    Returns
    -------
    joint_radii : (N,) per-node radii
    edge_radii : (E,2) per-edge start/end radii
    """
    from collections import defaultdict
    
    N = nodes.shape[0]
    E = edges.shape[0]

    # collect radii per node
    incident = defaultdict(list)
    for e, (i0, i1) in enumerate(edges):
        incident[i0].append(edge_mid_radii[e])
        incident[i1].append(edge_mid_radii[e])

    joint_radii = np.zeros(N, dtype=float)
    for i in range(N):
        if len(incident[i]) == 0:
            joint_radii[i] = 0.0
        elif len(incident[i]) == 1:
            # leaf: just use that edge's radius
            joint_radii[i] = incident[i][0]
        else:
            # average of all connected edge radii
            joint_radii[i] = np.mean(incident[i])

    # now expand to per-edge start/end
    edge_radii = np.zeros((E, 2), dtype=float)
    for e, (i0, i1) in enumerate(edges):
        edge_radii[e, 0] = joint_radii[i0]
        edge_radii[e, 1] = joint_radii[i1]

    return joint_radii, edge_radii

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

                # Look for "Components=" and check the subsequent character
                components_index = line.find("Components=")
                if components_index != -1:
                    start_index = components_index + len("Components=")
                    if start_index < len(line):  # Ensure there's a character after "Components="
                        subsequent_char = line[start_index]
                        if subsequent_char.isdigit():
                            # Create the key as a tuple (word, integer)
                            key = (words_between, int(subsequent_char))
                            # Add the tuple as a key to the dictionary with a placeholder value
                            results[key] = None  # Placeholder value; replace with desired value

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

def plot_binned_y_stats(xyz, values, n_bins=10, flip=False):
    """
    xyz:   (N, 3) array of x,y,z coordinates
    values: (N,) array of scalar values corresponding to each point
    n_bins: number of bins along the y-axis
    """

    # Normalise values 0-1
    values = (values - np.min(values)) / (np.max(values) - np.min(values))

    # Flip y-axis if requested
    if flip:
        xyz[:, 1] = -xyz[:, 1]

    y = xyz[:, 1]  # extract y-axis
    # Normalise y to 0-100%
    y = (y - y.min()) / (y.max() - y.min()) * 100
    bins = np.linspace(y.min(), y.max(), n_bins + 1)

    # digitize assigns each y to a bin index 1..n_bins
    bin_idx = np.digitize(y, bins) - 1

    means = []
    sds = []
    centers = []
    medians = []

    for i in range(n_bins):
        mask = bin_idx == i
        if np.any(mask):
            vals = values[mask]
            means.append(vals.mean())
            sds.append(vals.std())
            centers.append(0.5 * (bins[i] + bins[i+1]))
            medians.append(np.median(vals))
        else:
            means.append(np.nan)
            sds.append(np.nan)
            centers.append(0.5 * (bins[i] + bins[i+1]))

    means = np.array(means)
    sds = np.array(sds)
    centers = np.array(centers)

    plt.errorbar(means, centers, xerr=sds, fmt='o-', capsize=4)
    # plt.plot(medians, centers, 's--', label='Median')
    plt.ylabel("Lung height (Post→Ant)")
    plt.ylim(0,100)
    plt.xlabel("Value (mean ± SD)")
    plt.xlim(0,1.0)
    plt.title("Mean and SD of values across lung height (Post→Ant)")
    plt.grid(True)
    plt.show()

    return centers, means, sds

subject = 'EXAM5332'

# Read tree
nodes = extract_coordinates(f'results/{subject}/gradwarp/grown.ipnode')
edges = extract_global_numbers(f'results/{subject}/gradwarp/grown.ipelem')
radius = extract_radius(f'results/{subject}/gradwarp/grown_radius.ipfiel')
# Prepare tree info
edges = edges-1
radius, _ = compute_joint_radii(nodes, edges, radius)

dict_terminal = read_exnodedata(f'results/{subject}/gradwarp/terminal5.exnode') #term_average.exnode') # Read terminal.exnode
coords = np.array(dict_terminal[('coordinates',3)])
compliance = np.array(dict_terminal[('compliance',1)])
ppl = np.array(dict_terminal[('pleural pressure',1)]) # Pa
# resistance = np.array(dict_terminal[('resistance',1)])
vol_max = np.array(dict_terminal[('max volume',1)])
vol_tidal = np.array(dict_terminal[('tidal volume',1)])
flow = np.array(dict_terminal[('flow',1)])
vol_min = np.array(dict_terminal[('min volume',1)])
vol_tidal_total = sum(vol_tidal)
perc_vol_tidal = vol_tidal/vol_tidal_total*100
cluster_labels = kmeans_cluster(flow, n_clusters=5) +1 # 1-indexing
print(f"% Volume of VDP regions: {vol_max[cluster_labels==1].sum()/vol_max.sum():.2%} ") # % VDP
print(f"% Flow of VDP regions: {flow[cluster_labels==1].sum()/flow.sum():.2%} ") # % VDP
thresh = np.percentile(flow,15)
cluster_thresh = np.where(flow<thresh,1,2)
print(f"% Flow VDP thresh: {np.sum(cluster_thresh==1)/len(cluster_thresh):.2%}")
exit()
# # Read preful map
# # map_coords = np.load(f'../lobed_MRI_model/results/{subject}/vdpdefect_compressed.npz')['coordinates']
# map_coords = np.array(read_exnodedata(f'../lobed_MRI_model/results/{subject}/vdpmap_all.exnode')[('coordinates',3)])
# map_label = np.array(read_exnodedata(f'../lobed_MRI_model/results/{subject}/vdpmap_all.exnode')[('vdp label',1)])

# # Read lobe meshes
# import pyvista as pv
# lobes = ['RUL','RML','RLL','LUL','LLL']
# meshes = []
# for lobe in lobes:
    # meshes.append(pv.read(f'../lobed_MRI_model/results/{subject}/{lobe}.ply'))

# Read MoCoLoR Cluster 1 volume grid

ps.init()
ps.load_color_map('cluster_map','analysis/cluster_map.png')
tree = ps.register_curve_network('tree',nodes, edges,radius=0.015,color=[1,1,1])
tree.add_scalar_quantity('radius', radius)
tree.set_node_radius_quantity('radius',autoscale=False)

units = ps.register_point_cloud("units",coords,radius=0.015)
units.add_scalar_quantity('ppl',ppl,cmap='jet')
# units.add_scalar_quantity('resistance',resistance,cmap='jet')
units.add_scalar_quantity("tidal vol", vol_tidal,cmap='jet')
units.add_scalar_quantity("flow", flow,cmap='jet')
units.add_scalar_quantity("% tidal vol", perc_vol_tidal,enabled=True,cmap='jet',vminmax=(0,0.005))
units.add_scalar_quantity('min vol',vol_min,cmap='jet')
units.add_scalar_quantity('max vol',vol_max,cmap='jet')
units.add_scalar_quantity('compliance',compliance,cmap='jet')
units.add_scalar_quantity('cluster labels',cluster_labels,cmap='cluster_map')
units.add_scalar_quantity('max vol',vol_max,cmap='jet')
units.add_scalar_quantity('cluster thresh',cluster_thresh,cmap='jet')
# units.set_point_radius_quantity('% tidal vol')

# map = ps.register_point_cloud("preful map",map_coords,point_render_mode='quad',radius=0.0045)
# map.add_scalar_quantity("vdp label", map_label,enabled=True,cmap='jet',datatype='categorical')

# grp_mesh = ps.create_group("meshes")
# for (lobe,mesh) in zip(lobes,meshes):
#     vertices = mesh.points
#     if mesh.faces.size > 0:
#         face_array = mesh.faces.reshape((-1, 4))  # Assuming triangular faces
#         faces = face_array[:, 1:]  # shape: (N_faces, 3)
#     surf = ps.register_surface_mesh(f'{lobe}',vertices,faces, transparency=0.05,smooth_shade=True)
#     surf.add_to_group('meshes')
# grp_mesh.set_enabled(True)
# grp_mesh.set_show_child_details(False)

# Define a 3D volume using the Bounding Box of the 3D image in [X,Y,Z] format
import SimpleITK as sitk

img = sitk.ReadImage(f'../MoCoLoR/{subject}/struct_gradwarp/Images_mocolor_vent0.nii.gz')
img = sitk.Median(img,[3,3,3])
# Prepare image info
spacing = img.GetSpacing()
arr_img = sitk.GetArrayFromImage(img) # in [Z,Y,X]
shape = arr_img.shape # in (x,y,z)
swap_arr = arr_img.transpose(2,1,0) # rearrange [Z,Y,X] to [X,Y,Z]
img_block = ps.register_volume_grid("image block", (shape[2],shape[1],shape[0]),bound_low=(0,0,0),
                bound_high=(shape[2]*spacing[0],shape[1]*spacing[1],-shape[0]*spacing[2]),enabled=False)
# Add image intensity values to the 3D volume in [X,Y,Z] format
img_block.add_scalar_quantity("intensity",swap_arr,defined_on='nodes',cmap='gray',enabled=True)

cor_plane_pos = ps.add_scene_slice_plane()
cor_plane_pos.set_pose([0,0,0],[0,1,0])
cor_plane_pos.set_draw_widget(True)
cor_plane_pos.set_active(False)

cor_plane_neg = ps.add_scene_slice_plane()
cor_plane_neg.set_pose([0,0,0],[0,-1,0])
cor_plane_neg.set_draw_widget(True)
cor_plane_neg.set_active(False)

ax_plane_neg = ps.add_scene_slice_plane()
ax_plane_neg.set_pose([0,0,0],[0,0,-1])
ax_plane_neg.set_draw_widget(True)
ax_plane_neg.set_active(False)
ax_plane_pos = ps.add_scene_slice_plane()
ax_plane_pos.set_pose([0,0,0],[0,0,1])
ax_plane_pos.set_draw_widget(True)
ax_plane_pos.set_active(False)

ps.set_ground_plane_mode("none")
ps.set_navigation_style("free")
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([1,1,1,0])
ps.show()
