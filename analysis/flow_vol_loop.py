import numpy as np
import os, re
import matplotlib.pyplot as plt

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

subjects = ['vol669'] #['001','1C','3A','2C','EXAM5373','EXAM5332','EXAM5154']
duration = [4.0] #[2.403,2.35,2.327,3.01,4,4.045,5.404]
age = ['?'] #[9,10,8,13,22,47,25]
weight = [60] #[33,36,32,60,65,67,79]
n_phases = 10
fig, axs = plt.subplots(1,2,layout='constrained')
for i,subject in enumerate(subjects):
    flows = [0]
    # frc = np.sum(np.array(read_exnodedata(f'results/{subject}/gradwarp/term_average.exnode')[('min volume',1)]))/10**6
    frc = np.sum(np.array(read_exnodedata(f'../CHLA/{subject}/tree/filter_elems_in_ply/term_average.exnode')[('min volume',1)]))/10**6
    vols = [frc]
    for ph in range(2,n_phases+1):
        # flows.append(read_exelem(f'results/{subject}/gradwarp/terminal{ph}.exelem')[0])
        # vols.append(np.sum(np.array(read_exnodedata(f'results/{subject}/gradwarp/terminal{ph}.exnode')[('volume',1)]))/10**6)
        flows.append(read_exelem(f'../CHLA/{subject}/tree/filter_elems_in_ply/terminal{ph}.exelem')[0])
        vols.append(np.sum(np.array(read_exnodedata(f'../CHLA/{subject}/tree/filter_elems_in_ply/terminal{ph}.exnode')[('volume',1)]))/10**6)
        time = np.linspace(0,duration[i],n_phases+1)

    # region: OPTIONAL: interpolate more datapoints to make smooth curves === 
    # from scipy.interpolate import make_interp_spline
    # n_data = 30 # set num datspoints (more=smoother)

    # # 2. Create an "index tracker" t that strictly increases (0, 1, 2, 3...)
    # t = np.arange(len(vols))

    # # 3. Create a high-density version of our index tracker
    # t_smooth = np.linspace(0, len(vols) - 1, n_data)

    # # 4. Interpolate X and Y independently based on the index tracker
    # x_spline = make_interp_spline(t, np.array(vols), k=3)
    # y_spline = make_interp_spline(t, np.array(flows), k=3)

    # x_smooth = x_spline(t_smooth)
    # y_smooth = y_spline(t_smooth)

    # # 4. Assign new dataset
    # vols=list(x_smooth)
    # flows=list(y_smooth)
    # time = np.linspace(0,duration[i],n_data+1)
    # endregion: OPTIONAL: interpolate to make smooth curves === 
    
    flows.append(0) # close the loop
    vols.append(frc) # close the loop
    flows = np.array(flows)/10**6
    vols = np.array(vols)/10**6
    
    axs[0].plot(time, flows,label=f'{age[i]}Y; {weight[i]} kg')
    axs[0].set_xlabel("Time (s)",fontsize=16)
    axs[0].set_ylabel("Flow(t) (L/s)",fontsize=16)
    axs[0].grid()

    axs[1].plot(vols,flows)
    axs[1].set_xlabel("Volume (L)",fontsize=16)
    axs[1].set_ylabel("Flow(t) (L/s)",fontsize=16)
    axs[1].grid()
fig.legend(loc="outside lower center")
plt.show()
