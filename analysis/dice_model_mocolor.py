#!/usr/bin/env python3
import argparse, os, re
import numpy as np
import SimpleITK as sitk

def natural_key(s):
    """
    Split string into list of text and integer chunks for natural sorting.
    e.g. 'img10.nii.gz' -> ['img', 10, '.nii.gz']
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)
    ]

def read_model_results(dir):
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
    flow_t = []
    flow_proportion = []
    vol_t=[]
    vol_proportion=[]
    for f in exnode_files:
        dict_exnode = read_exnodedata(f,extn='.exnode')
        flow_t.append(dict_exnode[('flow',1)])
        fv = np.array(dict_exnode[('flow',1)])/np.sum(np.array(dict_exnode[('flow',1)]))
        flow_proportion.append(fv)
        vol_t.append(np.array(dict_exnode[('volume',1)]))
        vol_proportion.append(np.array(dict_exnode[('volume',1)])/np.sum(np.array(dict_exnode[('volume',1)])))
        
    coords=dict_exnode[('coordinates',3)]

    return {'vol_proportion':vol_proportion,'vol_t':vol_t,'flow_t_proportion':flow_proportion,'flow_t': np.array(flow_t), 'flow_t_trach': np.array(flow_t_trach), 'coords': np.array(coords)}

def read_imgs(dir,keyword):
    """
    Read images (background==-1) in the given directory.
    Return a list of SITK images.
    """
    nii_files = sorted(
        [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(".nii.gz") and keyword in f],
        key=natural_key
    )

    imgs = []
    for f in nii_files:
        img = sitk.ReadImage(f)
        imgs.append(img)
    
    print(f"Read {len(imgs)} files with keyword {keyword}")
    return imgs

def dice_score(mask1, mask2):
    """
    Compute DICE score for a specific label in two integer masks.
    DICE = 2 * |A ∩ B| / (|A| + |B|)
    """
    # 1. Binarise label 1
    m1 = np.where(mask1==1,1,0)
    m2 = np.where(mask2==1,1,0)

    # 2. Compute intersection and sizes
    intersection = (m1 * m2).sum()
    size1 = np.count_nonzero(m1)
    size2 = np.count_nonzero(m2)

    # 3. Handle empty case
    if size1 + size2 == 0:
        print(f"Empty in both masks → Dice = 1")
        return 1.0

    # 4. Compute DICE
    dsc = 2.0 * intersection / (size1 + size2)
    # print(f"Label {label} Dice Score Coefficient: {dsc:.6f}")
    return dsc

def dice_spatial_accuracy(mask1, mask2):
    """
        Dice Spatial Accuracy=DSC(Ventilated) + DSC(Ventilation Defect)
                            = 2*(mask1_vent ∩ mask2_vent)/(mask1_vent + mask2_vent) + 
                              2*(mask1_def ∩ mask2_def)/(mask1_def + mask2_def)
    """

    h1 = np.where(mask1>1,1,0)
    h2 = np.where(mask2>1,1,0)
    dsc_vent = dice_score(h1,h2)
    print(f"Ventilated Dice Score: {dsc_vent:.4f}")
    d1 = np.where(mask1==1,1,0)
    d2 = np.where(mask2==1,1,0)
    dsc_def = dice_score(d1,d2)
    print(f"Defect Dice Score: {dsc_def:.4f}")
    dsa = dsc_vent+dsc_def
    print(f"Dice Spatial Accuracy: {dsa:.4f}")
    return dsa

def combine_label1_regions(img1, img2):
    """
    Given two labelled SimpleITK images:
      - Keep img1 label 1 as label 1
      - Convert img2 label 1 to label 2
      - Voxels where both images have label 1 become label 3

    Returns a new sitk.Image with labels {0,1,2,3}.
    """

    # Boolean masks for label 1
    m1 = np.where(img1==1,1,0)
    m2 = np.where(img2==1,1,0)

    # Assign label 2 to m2
    m2 = np.where(m2==1,2,0)

    # Overlap mask
    overlap = m1+m2 # overlap region=3

    # Output image
    return overlap

def main():
    # outdir = os.path.join(model_folder, "dice-model-mocolor")
    # os.makedirs(outdir, exist_ok=True)
    
    # Compare phase-by-phase between mocolor Jac and "imaged" model calculations
    mocolor = read_imgs(mocolor_folder,keyword='hkm_clusters') # --- Read MoCoLoR clusters ---
    model = read_imgs(model_folder,keyword='kmeans_clusters') # --- Read MoCoLoR clusters ---
    
    # # Compare EI phase between mocolor Jac and max volumes from terminal solution
    # mocolor = 
    # model = read_model_results
    
    n_phases = len(model)
    for phase in range(n_phases):
        print(f"Phase {phase+2} ")
        moc = sitk.GetArrayFromImage(mocolor[phase])
        mod = sitk.GetArrayFromImage(model[phase])
        moc = np.where(moc==1,1,2) # 1:defect, 2: healthy
        mod = np.where(mod==1,1,2) # 1:defect, 2: healthy
        assert len(moc)==len(mod);"mismatched masks"
        dice = dice_score(moc, mod) # get dice score for ventilation defect cluster (label 1)
        # print(f"Dice score coefficient: {dice:.4f}")
        # dsa = dice_spatial_accuracy(moc,mod)

        combined = combine_label1_regions(moc,mod) # mocolor:1, model:2, combined:3
        combined=sitk.GetImageFromArray(combined)
        combined.CopyInformation(mocolor[0])
        sitk.WriteImage(combined,os.path.join(outdir,f'combined-VDP-cluster{phase+2}.nii.gz'))
        print(f'exported phase {phase+2}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical kmeans clustering on model and MoCoLoR ventilation distribution and evaluate DICE score for each cluster.")
    parser.add_argument("mocolor_folder", type=str, help="Path to subject folder containing MoCoLoR masked Jac clusters.")
    parser.add_argument("model_folder", type=str, help="Path to subject folder containing 'imaged' model clusters.")
    parser.add_argument("-outdir", type=str, help="Dir to combined label 1 image.")
    args = parser.parse_args()
    model_folder = args.model_folder # 'results/EXAM5332_10phases/gradwarp/'
    mocolor_folder = args.mocolor_folder # '../MoCoLoR/EXAM5332_10phases/jac_gradwarp/masked'
    outdir = args.outdir

    main()