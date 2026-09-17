import numpy as np
import ants
import pandas as pd # ants needs pd
import polyscope as ps
import polyscope.imgui as psim
import re
from collections import defaultdict

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

def myCallback(): # gets executed per-frame

  # Python scope resolution quirks
  # may need `nonlocal` rather than `global`` if your callback is 
  # defined inside another function
  global curr_frame, auto_playing

  update_frame_data = False
  _, auto_playing = psim.Checkbox("Autoplay", auto_playing)

  # Advance the frame
  if auto_playing:
    update_frame_data = True
    curr_frame = (curr_frame + 1) % n_phases

  # Slider to manually scrub through frames  
  slider_updated, curr_frame = psim.SliderInt("Curr Phase", curr_frame, 0, n_phases-1)
  update_frame_data = update_frame_data or slider_updated

  # Update the scene content if-needed
  if update_frame_data:
    # ps.register_point_cloud(f"phase {phase}", mapped_coords[curr_frame])
    tree=ps.register_curve_network(f"phase {phase}", mapped_coords[curr_frame],edges)
    tree.add_scalar_quantity('radius',radius)
    tree.set_node_radius_quantity('radius',False)

if __name__ == "__main__":
    #subject_trans = '../MoCoLoR/001_10phases' # Retrieve MoCoLoR transforms
    
    # Read in indices
    subject = 'EXAM5332_10phases'
    n_phases = 10
    data = np.load(f'results/{subject}/grown_idx.npz') # output of grow_lobes2, the final coords
    edges = extract_global_numbers(f'results/{subject}/grown.ipelem')
    edges = edges-1
    radius = extract_radius(f'results/{subject}/grown_radius.ipfiel')

    idx = pd.DataFrame(data['indices'],columns=['z','y','x']) # ants needs dataframe.
    spacing = data['spacing']
    coords = idx * spacing
    coords = coords.values[:,::-1] # ZYX to XYZ
    if not left_hand_system(coords): # to visualise in LH system
        coords = convert_left_right_system(coords)
    
    # ps.init()
    # radius, _ = compute_joint_radii(coords, edges, radius)
    # curr_frame = 0
    # auto_playing = False
    # # ps.register_point_cloud("original", coords) # ps can read dataframe too!
    # tree = ps.register_curve_network("original", coords,edges)
    # tree.add_scalar_quantity('radius',radius)
    # tree.set_node_radius_quantity('radius',False)
    
    # transform_fwd = ["1Warp.nii.gz", # non-linear warp from fixed to moving
    #                 "0GenericAffine.mat"] # affine

    transform_inv = ["0GenericAffine.mat", # affine inverse is inside the .mat
                    "1InverseWarp.nii.gz"] # inverse non-linear warp from moving to fixed

    idx = idx.rename(columns={'z':'x','y':'y','x':'z'}) # swap x and z indices to match transform's ZYX order
    mapped_indices = [idx] # store phase 1 indices (ZYX)
    mapped_coords = [coords] # store phase 1 coords (XYZ)
    for phase in range(2,n_phases+1): # skip phase 1
        prefix = f"../MoCoLoR/{subject}/transforms/phase_{phase}_" # prefix for the transform files
        print(f"Applying transforms for phase {phase}...")
        # t_fwd_list = [prefix + t_path for t_path in transform_fwd] # reset transform list for each phase
        t_inv_list = [prefix + t_path for t_path in transform_inv] # reset transform list for each phase

        # mapped_idx = ants.apply_transforms_to_points(dim=3, points=idx, transformlist=t_fwd_list)
        mapped_idx = ants.apply_transforms_to_points(dim=3, points=idx, transformlist=t_inv_list)
        mapped_indices.append(mapped_idx) # Save the mapped indices (ZYX)
        
        # Visualise points at the two phases
        coords = mapped_idx.values[:,::-1] * spacing # reorder ZYX to XYZ
        if not left_hand_system(coords): # to visualise in LH system
            coords = convert_left_right_system(coords)
        mapped_coords.append(coords)

    # Write tree nodes for each phase for sampling RV & reading into GAN
    np.savez_compressed(f'results/{subject}/transformed_trees.npz',indices=np.asarray(mapped_indices),coords=np.asarray(mapped_coords),spacing=spacing,edges=edges)

    # ps.set_user_callback(myCallback)
    # ps.set_background_color([0,0,0])
    # ps.set_ground_plane_mode("none")
    # ps.set_navigation_style("free")
    # ps.set_up_dir("z_up")
    # ps.set_front_dir("neg_y_front")
    # ps.show()
