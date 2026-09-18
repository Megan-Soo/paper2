#!/usr/bin/env python3
import os,re,argparse,tempfile,contextlib
import numpy as np
import SimpleITK as sitk
from aether.diagnostics import set_diagnostics_on
from aether.geometry import define_data_geometry, define_rad_from_geom, define_node_geometry, define_1d_elements, define_rad_from_file, list_tree_statistics
from aether.growtree import grow_tree, smooth_1d_tree
from aether.exports import export_1d_elem_field, export_node_geometry, export_1d_elem_geometry
from aether.indices import define_problem_type, get_ne_radius

def ex2ipnode(path, filename):
    ### Open .exnode file
    try:
        with open(os.path.join(path, filename + ".exnode")) as exnode:
            nodefile_data = exnode.readlines()
    except FileNotFoundError:
        print("NO " + filename + ".exnode EXISTS")
        return

    total_lines = len(nodefile_data)
    Xj_1 = {}
    Xj_2 = {}
    Xj_3 = {}

    ## Collect and store index of each node
    node_indices = [i for i, s in enumerate(nodefile_data) if 'Node' in s]

    ## Get total number of nodes
    total_nodes = len(node_indices)

    for i in range(len(node_indices)):
        # nodefile_data[i] = nodefile_data[i].rstrip('\n')
        Xj_1[i] = nodefile_data[node_indices[i] + 1]  # store Xj[1] of current node
        Xj_2[i] = nodefile_data[node_indices[i] + 2]  # store Xj[2] of current node
        Xj_3[i] = nodefile_data[node_indices[i] + 3]  # store Xj[3] of current node

    ### Create ipnode file
    with open(os.path.join(path, filename + '.ipnode'), 'w') as file:
        file.write(" CMISS Version 2.0  ipnode File Version 2\n")
        # file.write("Heading: {}\n".format())
        file.write(" Heading: \n")
        file.write("\n")
        file.write(" The number of nodes is [   {}]:    {}\n".format(total_nodes, total_nodes))
        file.write(" Number of coordinates [ 3]: 3\n")
        file.write(" Do you want prompting for different versions of nj=1 [N]? Y\n")
        file.write(" Do you want prompting for different versions of nj=2 [N]? Y\n")
        file.write(" Do you want prompting for different versions of nj=3 [N]? Y\n")
        file.write(" The number of derivatives for coordinate 1 is [0]: 0\n")
        file.write(" The number of derivatives for coordinate 2 is [0]: 0\n")
        file.write(" The number of derivatives for coordinate 3 is [0]: 0\n")
        file.write("\n")

        # Loop through each node
        for node in range(total_nodes):
            file.write(
                " Node number [    {}]:     {}\n".format(node + 1, node + 1))  # node+1 because of zero-indexing in python
            file.write(" The Xj(1) coordinate is [ 0.00000E+00]:  {}".format(Xj_1[node]))
            file.write(" The Xj(2) coordinate is [ 0.00000E+00]:  {}".format(Xj_2[node]))
            file.write(" The Xj(3) coordinate is [ 0.00000E+00]:  {}\n".format(Xj_3[node]))

def ex2ipelem(path, filename):
    ### Open .exelem file
    try:
        with open(os.path.join(path, filename + ".exelem")) as exelem:
            elemfile_data = exelem.readlines()
    except FileNotFoundError:
        print("NO " + filename + ".exelem EXISTS")
        exit()

    ## Get Group name
    group_name = elemfile_data[0][12:].rstrip('\n')  # get the string after "Group name: ", strip the "\n"

    ## Collect and store index of each element
    elem_indices = [i for i, s in enumerate(elemfile_data) if 'Element' in s]

    ## Get total number of elems
    total_elems = len(elem_indices)

    nodes_1 = {}

    for i in range(len(elem_indices)):
        nodes_1[i] = elemfile_data[elem_indices[i] + 2]  # the 1st #Nodes value in .exelem
        # nodes_2[i] = elemfile_data[elem_indices[i] + 3] # the 2nd #Nodes value in .exelem

    nj = 3  # following perl script terminology
    nb = 1  # following perl script terminology

    ### Create ipelem file
    with open(os.path.join(path, filename + '.ipelem'), 'w') as file:
        file.write(" CMISS Version 1.21  ipelem File Version 2\n")
        file.write(" Heading: {}\n".format(group_name))
        file.write("\n")
        file.write(" The number of elements is [{}]: {}\n".format(total_elems, total_elems))
        file.write("\n")
        for element in range(total_elems):
            file.write(" Element number [    {}]:    {}\n".format(element + 1,
                                                                element + 1))  # element+1 because of zero-indexing in python
            file.write(" The number of geometric Xj-coordinates is [{}]: {}\n".format(nj, nj))
            for i in range(3):
                file.write(" The basis function type for geometric variable {} is [{}]: {}\n".format(i + 1, nb, nb))
            file.write(" Enter the 2 global numbers for basis {}: {}\n".format(nb, nodes_1[element]))

def ex2ipfiel(path, filename):
    ### Open .exelem file
    try:
        with open(os.path.join(path, filename + ".exelem")) as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("NO " + filename + ".exelem EXISTS")
        exit()

    # Search for field value 'radius'
    found = False
    substring = 'radius'
    for line in lines:
        if substring.lower() in line.lower():
            found = True
            break
        else:
            found = False
            
    if found:
        ## Collect and store index of each element
        indices = [i for i, s in enumerate(lines) if 'Element' in s]

        # Regular expression to match numbers in scientific notation
        pattern = r"[-+]?\d*\.?\d+[eE][-+]?\d+"
        
        with open(os.path.join(path, filename+'.ipfiel'), 'w') as file:
            file.write(' CMISS Version 1.21 ipfiel File Version 3\n')
            file.write(' Heading: \n')
            file.write('\n')
            file.write(f' The number of elements is [  {len(indices)}]: {len(indices)}\n')
            for i in range(len(indices)):
                values = re.findall(pattern, lines[indices[i]+2])
                # Convert the matches to floats
                values = [float(num) for num in values]
                file.write('\n')
                file.write(f' Element number:     {i+1}\n')
                file.write(f' The field variable value is [ 0.00000D+00]: {values[-1]}\n')

    else:
        print(" Field value 'radius' not found in .exelem file. This function converts _radius.exelem files to .ipfiel.")

# === MAP DATAPOINT INITIAL VOLUMES TO TERMINAL NODES
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

def read_mapping_txt(file_path):
    with open(file_path, "r") as file:
        data = [line.split() for line in file.readlines()[1:]]  # Skip the first line (header)

    # Convert values to integers or floats where possible
    def convert(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value  # Keep as string if conversion fails

    data = [[convert(value) for value in row] for row in data]
    return data

def array2ipfiel_custom_number(path, filename, values, item_type='Element'): # for node ipfiel
    with open(os.path.join(path, filename + '.ipfiel'), 'w') as file:
        file.write(' CMISS Version 1.21 ipfiel File Version 3\n')
        file.write(' Heading: \n\n')
        file.write(f' The number of {item_type.lower()}s is [  {len(values)}]: {len(values)}\n')
        for i in range(len(values)):
            file.write(f'\n {item_type} number:        {values[i][0]}\n')
            file.write(f' The field variable value is [ 0.00000D+00]: {values[i][1]}')

def read_ipnode(file_path):
    ext = '.ipnode'
    try:
        with open(file_path + ext) as file_data:
            data = file_data.readlines()
    except FileNotFoundError:
        with open(file_path) as file_data:
            data = file_data.readlines()
    except FileNotFoundError as e:
        print(e)
        exit()
    
    coords = []

    ## Collect and store index of each node
    node_indices = [i for i, s in enumerate(data) if 'Node' in s]

    for i in range(len(node_indices)):
        x = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 1])[-1])
        y = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 2])[-1])
        z = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 3])[-1])
        node = [x,y,z]
        coords.append(node)  # append node coordinates

    return np.array(coords) # return list of coords

def coords_to_indices(coords_xyz, spacing):
    """
    Convert physical XYZ coordinates to nearest voxel indices
    given voxel spacing in ZYX order.

    Parameters
    ----------
    coords_xyz : (N, 3) array-like
        Physical coordinates in (x, y, z) order.
    spacing : (3,) array-like
        Voxel spacing in (x, y, z) order.

    Returns
    -------
    indices_zyx : (N, 3) ndarray of int
        Nearest voxel indices in (z, y, x) order.
    """
    coords_xyz = np.asarray(coords_xyz)
    spacing = np.asarray(spacing)

    # compute indices in xyz order
    # idx_xyz = np.round(coords_xyz / spacing).astype(int) # rounding it to integer changes geometry too much
    idx_xyz = coords_xyz / spacing # preserve float values

    # return indices in zyx order, coords in xyz order
    return idx_xyz[:, ::-1]

def main():
    parser = argparse.ArgumentParser(description="Grow tree and assign initial volume to terminal nodes. Generate.npz w/ nearest idx and spacing")
    parser.add_argument("-subject_dir", type=str, required=True, help="Assuming dir contains all required files (ipnode,ipelem,ipfiel,ipdata).")
    parser.add_argument("-outdir", type=str, required=False, help="DIR to save the grown files")
    parser.add_argument("-ref_img", type=str, required=False, help="Filepath to image volume to get spacing for .npz storing indices for mapping transforms.")
    args = parser.parse_args()

    f_im = args.ref_img
    subject_dir = args.subject_dir
    output_dir = args.outdir

    # If temp, use TemporaryDirectory(). If permanent, use nullcontext() which does nothing.
    is_temp = output_dir is None
    ctx = tempfile.TemporaryDirectory() if is_temp else contextlib.nullcontext()

    with ctx as temp_path:
        # Establish a single target directory path
        if is_temp:
            output_dir = os.path.abspath(temp_path)
            print(f"Writing to temp folder: {output_dir}")
        else:
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Writing to output directory: {output_dir}")

        subject = os.path.basename(subject_dir)
        print(f"Subject {subject}")

        set_diagnostics_on(False)
        define_problem_type('grow_tree')
        define_node_geometry(os.path.join(subject_dir, 'upper_airway'))
        define_1d_elements(os.path.join(subject_dir, 'upper_airway'))
        define_rad_from_file(os.path.join(subject_dir, 'upper_airway'))

        lobes = ['RUL','RML','RLL','LUL','LLL']
        for (lobe,parent) in zip(lobes,[10,12,13,8,9]):
            define_data_geometry(os.path.join(subject_dir,f'{lobe}.ipdata'))
            grow_tree([0],parent,0,60.0,20.0,0.4,1.0,1.2,180.0,True,os.path.join(output_dir,f'{lobe}_mapping'),'close')
            smooth_1d_tree(14,1.0) # smooth branches tt aren't image-derived. don't affect image-derived branches.

        order_system = 'fit'  # fit the radii between read-in values and min_rad at order 1
        start_at = 'inlet'    # required, but previous option should make obsolete?
        min_rad = 0.20         # radius of order 1 branches
        h_ratio = 0.0         # doesn't matter for the 'fit' option
        define_rad_from_geom(order_system, h_ratio, start_at, min_rad)

        filename = 'grown'
        group_name = '1d_tree'
        ne_radius = get_ne_radius()
        field_name = 'radius'
        export_1d_elem_field(ne_radius, os.path.join(output_dir, filename+'_radius'), group_name, field_name) # generate grown_radius.exelem
        ex2ipfiel(output_dir,filename+'_radius') # generate grown_radius.ipelem
        export_node_geometry(os.path.join(output_dir, filename), group_name) # generate grown.exnode
        ex2ipnode(output_dir,filename) # generate grown.ipnode
        export_1d_elem_geometry(os.path.join(output_dir, filename), group_name) # generate grown.exelem
        ex2ipelem(output_dir,filename) # generate grown.ipelem
        list_tree_statistics(os.path.join(output_dir,filename))

        # Map terminal units to their volumes
        paths_exdata = {f for f in os.listdir(subject_dir) if f.lower().endswith('.exdata')
                    and any(compartment.lower() in f.lower() for compartment in lobes)
                    }

        paths_mapping = {f for f in os.listdir(output_dir) if f.lower().endswith('.txt')
                        and any(compartment.lower() in f.lower() for compartment in lobes)
                        and 'mapping' in f.lower()}    

        if not paths_exdata:
            print(f'No .exdata found.')
            exit()
        elif not paths_mapping:
            print(f'No mapping txt found.')
            exit()

        exdata_dict = {compartment: next((f for f in paths_exdata if compartment.lower() in f.lower()), None)
                        for compartment in lobes}
        mapping_dict = {compartment: next((f for f in paths_mapping if compartment.lower() in f.lower()), None)
                        for compartment in lobes}

        init_vol_all = []
        term_elem_num_all = []
        coords_all = []
        term_node_num_all = []
        for compartment in lobes:
            path_exdata = os.path.join(subject_dir, exdata_dict.get(compartment))
            dict_exdata = read_exnodedata(path_exdata,extn='.exdata')
            init_vol = dict_exdata[('init vol',1)]

            # accumulate initial volumes of processed compartments
            init_vol_all = init_vol_all + init_vol
            coords_all = coords_all + dict_exdata[('coordinates',3)]

            # read {compartment}_mapping.txt
            path_mapping = os.path.join(output_dir, mapping_dict.get(compartment))
            mapping_list_2d = read_mapping_txt(path_mapping)
            # returns 2d list: [[datapoint no. (index of init_vol), Terminal element no., terminal unit no. (grown node number)]]
            
            # Sort the 2D list by the first column (datapoint col ie index of init_vol)
            sorted_data = sorted(mapping_list_2d, key=lambda x: x[0])

            # map terminal unit's correct elem number to its initial volume
            term_elem_num_all = term_elem_num_all + [row[1] for row in sorted_data]
            term_node_num_all = term_node_num_all + [row[2] for row in sorted_data]

        assert len(term_elem_num_all)==len(init_vol_all); "amount of Elem numbers and Initial volumes don't match"

        # Accumulate the 2d list like [[elem num1, init vol1],[elem num2, init vol2]]
        new_list = [[num,vol] for num,vol in zip(term_elem_num_all,init_vol_all)]

        # Sort the list by the first column (element number) in ascending order
        new_list = sorted(new_list, key=lambda x: x[0])

        # check total volume of terminal units
        total_volume = sum(init_vol_all)
        print(f"Total volume of terminal units: {total_volume/10**6:.2f} L")

        # write out new ipfiel file. to read into fortran w/ define_init_vol
        array2ipfiel_custom_number(output_dir,f'init_vol',new_list)

        if f_im:
            # ----------- Find nearest indices and save in .npz ------------
            coords = read_ipnode(os.path.join(output_dir,filename)) # Read grown.ipnode

            coords2=coords # make a copy
            if np.any(coords[:,2] < 0): # check if negative z coords (indicating that Z axis was flip to get LH coord system)
                print('flipping z axis...')
                coords2[:, 2] = -coords[:, 2]# flip back to positive z before getting indices
            
            img = sitk.ReadImage(f_im) # Read image

            idx = coords_to_indices(coords2,img.GetSpacing()) # Get indices (ZYX) float values

            np.savez_compressed(os.path.join(output_dir,'grown_idx.npz'), indices=idx, spacing=img.GetSpacing()) # Save grown_nearest_idx.npz (for transform application)

        # region: code for visualise_pipeline
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

        coords_upp = read_ipnode(os.path.join(subject_dir,'upper_airway.ipnode'))
        edges_upp = extract_global_numbers(os.path.join(subject_dir,'upper_airway.ipelem'))
        radius_upp = extract_radius(os.path.join(subject_dir,'upper_airway.ipfiel'))
        # fill in zero values for unassigned elements
        radius_upp = np.pad(radius_upp,(0,len(edges_upp)-len(radius_upp)))
        edges_upp=edges_upp-1

        coords = read_ipnode(os.path.join(output_dir,'grown.ipnode')) # Read grown.ipnode
        edges = extract_global_numbers(os.path.join(output_dir,'grown.ipelem'))
        radius = extract_radius(os.path.join(output_dir,'grown_radius.ipfiel'))
        edges = edges-1
        radius, _ = compute_joint_radii(coords, edges, radius)
        node_indices = np.array(term_node_num_all) -1 # switch to zero-indexing
        coords_terminal = coords[node_indices,:]
        
        steps= {"1. Upper airway centreline":"From get_centreline.py",
                "2. Grown tree":"Grown from upper_airway centreline.",
                "3. Acini units":"Spatial and volume distribution derived from generate_tissue_units.py. \
                    Note that the tree's terminal branches don't map exactly to the generated acini units.",
                "4. Final airway model":"Map volume distribution from generate_tissue_units.py to terminal branches."
                }

        np.savez_compressed('visualise_pipeline/grow_lobes_001.npz',
                            steps=steps,
                            step1_1a=coords_upp,
                            step1_1b=edges_upp,
                            step1_1c=radius_upp,
                            step2_1a=coords,
                            step2_1b=edges,
                            step2_1c=radius,
                            step3=coords_all,
                            step4_1a=coords_terminal,
                            step4_1b=init_vol_all,
                            )
        # endregion

if __name__ == "__main__":
    main()
