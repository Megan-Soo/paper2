from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import argparse,os,re
import pandas as pd
import SimpleITK as sitk
import pyvista as pv # need to peel the density image

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

def erode_image(image, radius=1, out=None):
    """
    Erode all voxels where image != -1.
    Voxels equal to -1 remain untouched.
    """
    # 1. Create binary mask of all voxels != -1
    mask = sitk.BinaryThreshold(
        image,
        lowerThreshold=-0.9,   # everything > -1
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0
    )

    # 2. Erode the mask
    eroder = sitk.BinaryErodeImageFilter()
    eroder.SetKernelRadius([radius, radius, radius])
    eroder.SetForegroundValue(1)
    eroded_mask = eroder.Execute(mask)

    # 3. Apply eroded mask back to original image
    result = sitk.Mask(image, eroded_mask, outsideValue=-1)

    if out is not None:
        sitk.WriteImage(result, out)

    return result

def bin_rv(image, n_bins, direction=1):
    # Assumes background of -1. Idky if set bg as user input, the result changes...

    # Move the chosen direction to the first axis
    image = np.moveaxis(image, direction, 0)  # shape: (slices, ..., ...)

    # Keep only slices without -1 values
    nonzero_mask = np.any(image != -1, axis=(1, 2))
    image = image[nonzero_mask]

    # Split into bins along the first axis
    img_bins = np.array_split(image, n_bins)

    # Compute mean and std of non -1 values in each bin
    means = np.array([bin_[bin_ != -1].mean() if np.any(bin_ != -1) else np.nan for bin_ in img_bins])
    stds = np.array([bin_[bin_ != -1].std() if np.any(bin_ != -1) else np.nan for bin_ in img_bins])

    return means, stds

def bin_rv_proportions(image, n_bins, direction=1):
    # Assumes background of -1. Idky if set bg as user input, the result changes...

    # Move the chosen direction to the first axis
    image = np.moveaxis(image, direction, 0)  # shape: (slices, ..., ...)

    # Keep only slices without -1 values
    nonzero_mask = np.any(image != -1, axis=(1, 2))
    image = image[nonzero_mask]

    # Split into bins along the first axis
    img_bins = np.array_split(image, n_bins)

    # Total sum of non -1 values in the entire image
    total_vol = np.sum(image[image != -1])

    # Compute mean and std of non -1 values in each bin
    proportions = np.array([bin_[bin_ != -1].sum()/total_vol if np.any(bin_ != -1) else np.nan for bin_ in img_bins])

    return proportions

def bin_image_density(image, n_bins, direction=1):

    # Move the chosen direction to the first axis
    image = np.moveaxis(image, direction, 0)  # shape: (slices, ..., ...)

    # Keep only slices with any non-zero values
    nonzero_mask = np.any(image != 0, axis=(1, 2))
    image = image[nonzero_mask]

    # Split into bins along the first axis
    img_bins = np.array_split(image, n_bins)

    # Compute mean and std of non-zero values in each bin
    means = np.array([bin_[bin_ != 0].mean() if np.any(bin_ != 0) else np.nan for bin_ in img_bins])
    stds = np.array([bin_[bin_ != 0].std() if np.any(bin_ != 0) else np.nan for bin_ in img_bins])

    return means, stds

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

def read_trees(path):
    try:
        data = np.load(path)
    except Exception as e:
        print(e)
        exit()

    edges = data['edges'] # (n_elems, 2)
    # rv_t = data['signals']
    coords = data['coords']

    return edges, coords

def main():
    n_phases = 10 
    animate = True
    height_bins = 20

    terminal_dict = read_exnodedata(os.path.join(model_folder,'term_average.exnode')) # Read terminal.exnode

    coords = np.array(terminal_dict[('coordinates',3)]) # shape (n_nodes, 3)
    z_max = np.max(coords[:,1])
    z_min = np.min(coords[:,1])
    # % lung height from posterior to anterior
    lung_height = z_max - z_min
    perc_height = (coords[:,1] - z_min) / lung_height * 100
    perc_height = 100 - perc_height # flip: smallest y (posterior): 0%, largest y (anterior): 100%
    print(f'Lung Height (mm): {lung_height:.2f}')

    if not animate:
        for phase in range(2,n_phases): #//2+1):
            terminal_dict = read_exnodedata(f'results/{subject}/gradwarp/terminal{phase}.exnode') # Read terminal.exnode

            z_max = np.max(coords[:,1])
            z_min = np.min(coords[:,1])
            # % lung height from posterior to anterior
            lung_height = z_max - z_min
            perc_height = (coords[:,1] - z_min) / lung_height * 100
            perc_height = 100 - perc_height # flip: smallest y (posterior): 0%, largest y (anterior): 100%
            print(f'Subject {subject}, Lung Height (mm): {lung_height:.2f}')

            trach = read_exelem(f'results/{subject}/gradwarp/terminal{phase}.exelem')[0] # Read terminal.exelem
            
            # flow
            vent = np.array(terminal_dict[('flow',1)])
            total_vent = np.sum(vent)

            # tidal volume
            tidal_vol = np.array(terminal_dict[('tidal volume',1)])
            total_tidal = np.sum(tidal_vol)

            # max volume
            max_vol = np.array(terminal_dict[('max volume',1)])
            total_vol = np.sum(max_vol)

            # use pandas to bin data
            df = pd.DataFrame({
                "perc_height": perc_height,
                "vent": vent,
                "tidal_vol": tidal_vol,
                "max_vol":max_vol
            })

            df["bin"] = pd.cut(df["perc_height"], bins=height_bins)

            # 2. Compute mean and SD per bin
            stats_vent = df.groupby("bin",observed=False)["vent"].agg(["mean", "std","sum"]).reset_index()
            stats_vent_list.append(stats_vent["sum"].values/total_vent)

            stats_tidal = df.groupby("bin",observed=False)["tidal_vol"].agg(["mean", "std","sum"]).reset_index()
            stats_tidal_list.append(stats_tidal["sum"].values/total_tidal)

            stats_vol = df.groupby("bin",observed=False)["max_vol"].agg(["sum"]).reset_index()
            stats_vol_list.append(stats_vol["sum"].values/total_vol)

    else: # animate phases
        bin_edges = np.linspace(0, 100, height_bins + 1)       # 0,10,20,...,100
        fig, ax =plt.subplots()
        stats_vent_list=[]
        stats_tidal_list=[]
        stats_vol_list=[]
        proportion_mocolor_list=[]
        proportion_model_list=[]
        for phase in range(2,n_phases+1): #//2+1):

            # Read MoCoLoR jac image
            mocolor = sitk.ReadImage(os.path.join(mocolor_folder,f'jac_gradwarp_phase_{phase}.nii.gz')) # should be RAS
            # mocolor = sitk.ReadImage(f'../MoCoLoR/{subject}/jac_gradwarp/masked/jac_gradwarp_phase_{phase}.nii.gz') # should be RAS
            # mc = mocolor
            # mocolor = sitk.Median(mocolor,[3,3,3])

            # bin MoCoLoR jac image
            # normalise ei to 0-1 range, ignoring background -1 values
            ei = sitk.GetArrayFromImage(mocolor)
            ei = np.flip(ei, axis=1) # flip to match posterior 0% anterior 100%
            bg = -1
            mocolor = np.where(ei == bg, ei, (ei - ei[ei != bg].min()) /
                                    (ei[ei != bg].max() - ei[ei != bg].min()))

            proportion_mocolor_list.append(bin_rv_proportions(mocolor,height_bins)) # these values sum to 1

            # Read model image
            model = sitk.ReadImage(os.path.join(model_folder,f'model-to-img/flow{phase}.nii.gz'))
            # model =sitk.ReadImage(f'results/{subject}/gradwarp/model-to-img/flow{phase}.nii.gz')
            # normalise ei to 0-1 range, ignoring background -1 values
            ei = sitk.GetArrayFromImage(model)
            ei = np.flip(ei, axis=1) # flip to match posterior 0% anterior 100%
            bg = -1
            model = np.where(ei == bg, ei, (ei - ei[ei != bg].min()) /
                                    (ei[ei != bg].max() - ei[ei != bg].min()))            
            proportion_model_list.append(bin_rv_proportions(model,height_bins))

        # store artists so we can remove them
        hist_artists = []
        # Create text annotation once
        text = ax.text(0.1,95, "", transform=ax.transAxes)
        writer = PillowWriter(fps=1)
        # outfile = f"results/{subject}/gradwarp/flow-img-mocolor-phases-no-dilate.gif"
        outfile = os.path.join(outdir,'fv.gif')
        count=0
        with writer.saving(fig, outfile, dpi=150):
            while True:
                count+=1
                # stop the script if the plot window is closed
                if not plt.fignum_exists(fig.number):
                    break
                for t in range(len(proportion_mocolor_list)):
                    # proportion_vent = stats_vent_list[t]
                    # proportion_tidal = stats_tidal_list[t]
                    # proportion_vol = stats_vol_list[t]
                    proportion_mocolor = proportion_mocolor_list[t]
                    proportion_model = proportion_model_list[t]

                    # remove previous histogram
                    for artist in hist_artists:
                        artist.remove()
                    hist_artists=[]

                    # # draw model histogram
                    # phase, bins, patches0 = ax.hist(
                    #     bin_edges[:-1], bin_edges,
                    #     weights=proportion_vent,
                    #     orientation='horizontal',
                    #     alpha=0.5,
                    #     edgecolor='blue',
                    #     label='Model',
                    #     histtype='step',
                    #     linewidth=3
                    # )

                    # # draw model histogram
                    # phase, bins, patches1 = ax.hist(
                    #     bin_edges[:-1], bin_edges,
                    #     weights=proportion_vol,
                    #     orientation='horizontal',
                    #     alpha=1.0,
                    #     edgecolor='orange',
                    #     label='Model',
                    #     histtype='step',
                    #     linewidth=3
                    # )

                    # draw mocolor histogram
                    phase, bins, patches2 = ax.hist(
                        bin_edges[:-1], bin_edges,
                        weights=proportion_mocolor,
                        orientation='horizontal',
                        alpha=0.5,
                        color='green',
                        label='MoCoLoR'
                    )

                    # draw model histogram
                    phase, bins, patches3 = ax.hist(
                        bin_edges[:-1], bin_edges,
                        weights=proportion_model,
                        orientation='horizontal',
                        alpha=0.5,
                        color='black',
                        label='Model'
                    )

                    # text.set_text(f"Q(t): {round(trach/10**3)} mL/s")
                    ax.set_title(f"Phase {t+2}")
                    ax.legend(['MoCoLoR','Flow(t)'])
                    ax.set_xlabel('Fractional Ventilation',fontsize=16)
                    ax.set_ylabel('Posterior to Anterior (%)',fontsize=16)
                    ax.set_xlim(0,0.2)

                    # collect artists to remove next frame
                    # hist_artists.extend(patches0)                 # rectangles
                    # hist_artists.extend(patches1)                 # rectangles
                    hist_artists.extend(patches2)                 # rectangles
                    hist_artists.extend(patches3)                 # rectangles
                    hist_artists.extend(ax.lines[-1:])           # the step outline line

                    # Save frame
                    writer.grab_frame()
                    if count==10:
                        exit()

                    plt.pause(0.5)
        # exit()

    # fig.supxlabel('Normalised Acinar FRC volume or Tissue density (0-1)',fontsize=16)
    # fig.supxlabel('Normalised Values',fontsize=16)
    fig.supxlabel('Fractional Ventilation',fontsize=16)
    fig.supylabel('Posterior to Anterior (%)',fontsize=16)
    if n_phases>1:
        fig.legend(['Flow(t)','Tidal Volume','MoCoLoR'])
    else:
        fig.legend(['Posterior lung','Anterior lung'])
    # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical kmeans clustering on model and MoCoLoR ventilation distribution and evaluate DICE score for each cluster.")
    parser.add_argument("model_folder", type=str, help="Path to subject folder containing model ventilation results.")
    parser.add_argument("mocolor_folder", type=str, help="Path to subject folder containing MoCoLoR masked Jac clusters.")
    parser.add_argument("-outdir", type=str, help="Dir to combined label 1 image.")
    args = parser.parse_args()
    model_folder = args.model_folder # 'results/EXAM5332_10phases/gradwarp/'
    mocolor_folder = args.mocolor_folder # '../MoCoLoR/EXAM5332_10phases/jac_gradwarp/masked'
    outdir = args.outdir

    main()
