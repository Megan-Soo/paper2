"""
Calculate fractional ventilation of each lobe
"""
import os, re
import numpy as np

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

subjects = ['001_10phases']
fv_all = []
for s in subjects:  # For each subject,
    # Read terminal
    terminal = read_exnodedata(f'results/{s}/terminal.exnode')
    node_num = np.array(terminal[('Node number', None)])
    flow = np.array(terminal[('flow',1)])
    total_flow = np.sum(flow)
    # Read tree connections
    tree_connections = extract_global_numbers(f'results/{s}/grown.ipelem')

    fv = []
    for (pe, lobe) in [(10, 'RUL'), (12, 'RML'), (13, 'RLL'), (8, 'LUL'), (9, 'LLL')]: # For each parent elem and corresponding lobe
        terminals = np.array(read_mapping_txt(f'results/{s}/{lobe}_mapping.txt'))[:,2]
        lobe_flow = np.sum([flow[node_num==t][0] for t in terminals]) # Sum flow of terminal nodes in this subtree    
        fv = fv + [lobe_flow/total_flow] # Fractional ventilation of this lobe
        # Fractional ventilation per lobe
        print(f"{lobe} fractional ventilation: {fv[-1]*100:.2f}%")# Terminal nodes downstream of a parent elem

    fv_all.append(fv)

# Mean SD of ventilation per lobe for all subjects