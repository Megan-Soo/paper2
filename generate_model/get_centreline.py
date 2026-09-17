#!/usr/bin/env python3
import os, shutil, argparse
import SimpleITK as sitk
import centreline_gui
import numpy as np

def array2exnode(path, filename, nodes):
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes)

    with open(os.path.join(path, filename + '.exnode'), 'w') as file:
        file.write(" Group name: \n")
        file.write(" !#nodeset nodes\n")
        file.write(' #Fields=1\n')
        file.write(" 1) coordinates, coordinate, rectangular cartesian, #Components=3\n")
        file.write("  x.  Value index= 1, #Derivatives=0, #Versions=1\n")
        file.write("  y.  Value index= 2, #Derivatives=0, #Versions=1\n")
        file.write("  z.  Value index= 3, #Derivatives=0, #Versions=1\n")

        for i in range(len(nodes)):
            file.write(f" Node:            {i+1}\n")
            # line = str(nodes[i]).strip("[]")
            file.write(f"  {nodes[i][0]}\n")
            file.write(f"  {nodes[i][1]}\n")
            file.write(f"  {nodes[i][2]}\n")

def array2ipnode(path, filename, nodes):
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes)

    with open(os.path.join(path, filename + '.ipnode'), 'w') as file:
        file.write(' CMISS Version 2.0  ipnode File Version 2\n')
        file.write(' Heading: \n')
        file.write('\n')
        file.write(f' The number of nodes is [   {len(nodes)}]:    {len(nodes)}\n')
        file.write(' Number of coordinates [ 3]: 3\n')
        file.write(' Do you want prompting for different versions of nj=1 [N]? Y\n')
        file.write(' Do you want prompting for different versions of nj=2 [N]? Y\n')
        file.write(' Do you want prompting for different versions of nj=3 [N]? Y\n')
        file.write(' The number of derivatives for coordinate 1 is [0]: 0\n')
        file.write(' The number of derivatives for coordinate 2 is [0]: 0\n')
        file.write(' The number of derivatives for coordinate 3 is [0]: 0\n')
        file.write('\n')
        for i in range(len(nodes)):
            file.write(f' Node number [    {i+1}]:     {i+1}\n')
            file.write(f' The Xj(1) coordinate is [ 0.00000E+00]:    {nodes[i][0]}\n')
            file.write(f' The Xj(2) coordinate is [ 0.00000E+00]:    {nodes[i][1]}\n')
            file.write(f' The Xj(3) coordinate is [ 0.00000E+00]:    {nodes[i][2]}\n')
            file.write('\n')

def read_ipnode(file_path):
    ext = '.ipnode'
    try:
        with open(file_path + ext) as file_data:
            data = file_data.readlines()
    except FileNotFoundError as e:
        print(" NO " + file_path + ext + " EXISTS")
    
    coords = []

    ## Collect and store index of each node
    node_indices = [i for i, s in enumerate(data) if 'Node' in s]

    for i in range(len(node_indices)):
        x = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 1])[-1])
        y = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 2])[-1])
        z = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 3])[-1])
        node = [x,y,z]
        coords.append(node)  # append node coordinates

    return coords # return list of coords

def write_ipelem_file(filename):
    """
    Generate a CMISS .ipelem file with the predefined 13-element 1D tree.
    """
    # --- predefined element connections ---
    elem_nodes = [
        (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (5, 8), (7, 9),
        (7,10), (8,11), (8,12), (12,13),
        (12,14)
    ]

    with open(filename, "w") as f:
        f.write(" CMISS Version 1.21  ipelem File Version 2\n")
        f.write(" Heading:  1d_tree\n\n")
        f.write(f" The number of elements is [{len(elem_nodes)}]: {len(elem_nodes)}\n\n")

        for i, (n1, n2) in enumerate(elem_nodes, start=1):
            f.write(f" Element number [    {i}]:    {i}\n")
            f.write(" The number of geometric Xj-coordinates is [3]: 3\n")
            f.write(" The basis function type for geometric variable 1 is [1]: 1\n")
            f.write(" The basis function type for geometric variable 2 is [1]: 1\n")
            f.write(" The basis function type for geometric variable 3 is [1]: 1\n")
            f.write(f" Enter the 2 global numbers for basis 1:  {n1} {n2}\n\n")

    print(f"Generated {filename}", end='')

def visualise_upper(nodes,img):
    """
    Takes in 14 node coordinates and visualises with predefined edge connections.
    """
    import polyscope as ps
    import numpy as np
    edges = np.array([
        (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (5, 8), (7, 9),
        (7,10), (8,11), (8,12), (12,13),
        (12,14)
    ]) - 1

    nodes = np.asarray(nodes)

    ps.init()
    ps.register_curve_network("upper", nodes, edges)
    # Define a 3D volume using the Bounding Box of the 3D image in [X,Y,Z] format
    # Prepare image info
    spacing = img.GetSpacing()
    arr_img = sitk.GetArrayFromImage(img) # in [Z,Y,X]
    shape = arr_img.shape # in (x,y,z)
    swap_arr = arr_img.transpose(2,1,0) # rearrange [Z,Y,X] to [X,Y,Z]
    img_block = ps.register_volume_grid("image block", (shape[2],shape[1],shape[0]),bound_low=(0,0,0),
                    bound_high=(shape[2]*spacing[0],shape[1]*spacing[1],-shape[0]*spacing[2]),enabled=True)
    # Add image intensity values to the 3D volume in [X,Y,Z] format
    img_block.add_scalar_quantity("intensity",swap_arr,defined_on='nodes',cmap='gray',enabled=True)
    
    ps.set_background_color([0,0,0])
    ps.set_ground_plane_mode('none')
    ps.set_navigation_style('free')
    ps.show()

def main():
    parser = argparse.ArgumentParser(description="Run GUI to select upper airway and upper artery centreline points from 3D image.")
    parser.add_argument("-img", required=True, help="Path to image file (e.g. image.nii.gz)")
    parser.add_argument("-outdir", required=True, help="DIR to save the upper airway.ipnode/ipelem files.")
    parser.add_argument("-get_upper_airway", type=str, default=1, help="Run GUI to select upper airway centreline points.")
    parser.add_argument("-get_upper_artery", type=str, default=0, help="Run GUI to select upper artery centreline points. Select trunk, mid-trunk,trunk bif, LPA, RPA.")
    parser.add_argument("-view", default='c', choices=['a', 's', 'c'], help="Initial view for GUI: axial, sagittal, coronal (default: coronal)")
    args = parser.parse_args()

    output_directory = args.outdir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    img_orientation = 'LPI'
    
    image = sitk.ReadImage(args.img)
    gui_view = args.view

    if args.get_upper_airway==1:
        upper_airway_centreline = centreline_gui.GetUpperAirway(image, img_orientation, gui_view).centreline
        filename = 'upper_airway'
        array2exnode(output_directory, filename, upper_airway_centreline)
        array2ipnode(output_directory, filename, upper_airway_centreline)
        if not os.path.exists(os.path.join(output_directory,'upper_airway.ipelem')):
            write_ipelem_file(os.path.join(output_directory,'upper_airway.ipelem')) # generate 14-node upper airway template ipelem file
        visualise_upper(upper_airway_centreline,image)

    if args.get_upper_artery==1: # upper airway centreline is a prerequisite

        print(' Searching for existing upper_airway ipnode...')
        try:
            upper_airway_centreline = read_ipnode(os.path.join(output_directory, 'upper_airway'))
            print(' Found existing upper_airway.ipnode. Upper artery will replace Nodes 1-4 of upper_airway.')
        except:
            print(' No existing upper_airway.ipnode found.')

        print(' Searching for existing grown airway ipnode...')
        try:
            grown_airway = read_ipnode(os.path.join(output_directory, 'grown'))
            grown_exists = True
            print(' Found existing grown.ipnode. Upper artery centreline will be attached to this tree.')
        except:
            grown_exists = False
            print(' No existing grown.ipnode found.')

        try: # select upper artery points and edit upper_airway_centreline nodes
            upper_artery_centreline = centreline_gui.GetUpperArtery(image, img_orientation, gui_view, upper_airway_centreline).centreline
            filename = 'upper_artery'
            array2exnode(output_directory, filename, upper_artery_centreline)
            array2ipnode(output_directory, filename, upper_artery_centreline)
        except:
            print(' Cancelled get_upper_artery.')
            exit()

        if not os.path.exists(os.path.join(output_directory,'upper_artery.ipelem')):
            write_ipelem_file(os.path.join(output_directory,'upper_artery.ipelem')) # generate 14-node upper artery template ipelem file

        if grown_exists:
            import generate
            print(' Attaching derived upper_artery to grown airway...')
            artery_full = generate.attach_artery_to_grown_airway(upper_artery_centreline, grown_airway)
            filename = 'artery_full'
            array2exnode(output_directory, filename, artery_full)
            array2ipnode(output_directory, filename, artery_full)
            shutil.copy(os.path.join(output_directory, 'grown.exelem'), os.path.join(output_directory,'artery_full.exelem'))
            shutil.copy(os.path.join(output_directory, 'grown.ipelem'), os.path.join(output_directory,'artery_full.ipelem'))

if __name__ == "__main__":
    main()