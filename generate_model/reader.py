import SimpleITK as sitk
# from scipy.io import loadmat
from loguru import logger
import os, dynamic_yaml
import re

class Reader:
    # Reader will: (using sitk package)
    # 1. detect image file type & read accordingly (DICOM needs extra commands)
    # Optional: if masks are stored in same folder, use keyword to read the correct mask

    def __init__(self, data_folder, keyword=False):
        self.data_folder = data_folder
        # collect all filenames. skip hidden files (startswith '.')
        self.files = [f for f in os.listdir(self.data_folder) if not f.startswith('.')]
        self.keyword = keyword
        self.read()

    def read(self):
        try:
            self.image = self.dicom_reader()
        except:
            self.image = self.file_reader()

    # if single image file
    def file_reader(self):
        if self.keyword is not False:
            # filter out files with keyword
            self.files = [s for s in self.files if self.keyword.lower() in s.lower()]

        if not self.files:
            logger.debug(f' File with keyword \'{self.keyword}\' not found.')
            # logger.warning(' WARNING: An error will be thrown if this file is required in subsequent image processing tasks.')
            return False

        for f in self.files:
            try:
                image = sitk.ReadImage(os.path.join(self.data_folder,f))
                image = sitk.DICOMOrient(image,'LPI')
            except:
                continue

        return image

    # if DICOM series
    def dicom_reader(self):
        sitk.ProcessObject_SetGlobalWarningDisplay(False)  # warning is written in C++, can't use Py Warnings mod to suppress
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(self.data_folder)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            image = sitk.DICOMOrient(image,'LPI')
        except:
            pass
        sitk.ProcessObject_SetGlobalWarningDisplay(True)  # reinstate warnings

        return image
    
    def image_to_ras_array(self, obj):
        try:
            array = sitk.GetArrayFromImage(sitk.DICOMOrient(obj, 'LPI'))
        except:
            logger.error(f' Error getting RASArray.')

        return array

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

    def read_ipelem(path):
        elements = {}
        with open(path, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Element number"):
                # Get element number
                element_number = int(line.split(":")[1].strip())
                # Advance 5 lines to reach the line with the 2 global node numbers
                i += 5
                node_line = lines[i].strip()
                # Extract node numbers
                node_nums = list(map(int, node_line.split(":")[1].strip().split()))
                elements[element_number] = node_nums
            i += 1

        return elements

    def read_exdata(file_path):

        try:
            file_path, ext = os.path.splitext(file_path)
            if bool(ext) is False:
                ext = '.exdata'

            with open(file_path + ext) as file_data:
                data = file_data.readlines()
            
            coords = []

            ## Collect and store index of each node
            node_indices = [i for i, s in enumerate(data) if 'Node' in s]

            try:
                for i in range(len(node_indices)): # if coords written in rows instead of columns
                    x = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 1])[0])
                    y = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 1])[1])
                    z = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 1])[2])
                    node = [x,y,z]
                    coords.append(node)  # append node coordinates

            
            except: # if coords written in rows instead of cols
                for i in range(len(node_indices)): # if coords written in rows instead of columns
                    x = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 1])[0])
                    y = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 2])[0])
                    z = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[node_indices[i] + 3])[0])
                    node = [x,y,z]
                    coords.append(node)  # append node coordinates
        
            return coords # return list of coords
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            exit()

    def read_ipdata(filename):
        """
        Reads a text file with the format:
        index x y z 1.0 1.0 1.0
        and returns a NumPy array (n, 3) of x, y, z coordinates.
        
        :param filename: Path to the text file
        :return: NumPy array of shape (n, 3)
        """
        data = []
        with open(filename, 'r') as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])  # Extract x, y, z
                data.append([x, y, z])
        
        return data

    def read_ipfiel(file_path,item_type='Element'):
        extn = '.ipfiel'
        try:
            file_path, ext = os.path.splitext(file_path)
            if bool(ext) is False:
                ext = extn
            
            with open((file_path+ext), 'r') as file:
                data= file.readlines()
            
            values = []
            numbers = []
            ## Collect and store value of each element
            indices = [i for i, s in enumerate(data) if item_type in s]

            for i in range(len(indices)):
                number = int(re.findall(r"\b-?\d+\b",data[indices[i]])[-1]) # get Node/Element number
                numbers.append(number)
                val = float(re.findall(r"[-+]?(?:\d*\.*\d+)", data[indices[i] + 1])[-1]) # get field value
                values.append(val)

            my_dict = {
                        (f'{item_type} number',None): numbers,
                        ('values',1): values
                    }

            return my_dict
        
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            exit()
    
    def read_bin_txt(filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        my_dict = {}
        for line in lines:
            parts = line.split(';') # Split the string into parts separated by spaces

            for part in parts:
                    # Split by colon
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Check if multiple entries or single entry
                        array = [x for x in value.split(",")]
                        if len(array)>1:
                            # assuming arrays only have numbers, convert to float values
                            array = [float(x) for x in array]
                            my_dict[key]=array # get array from the string
                        else:
                            value = array[0] # extract the single value
                            # Detect if the value is a float or a word
                            if value.replace('.', '', 1).isdigit():
                                my_dict[key]=float(value)
                            elif value.isalpha():
                                my_dict[key]=value
                            else:
                                my_dict[key]=None  # If it's neither a word nor a float

        return my_dict

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
    
    def load_config(config_path):
        # Load the configuration file using PyYAML
        with open(config_path, 'r') as f:
            cfg = dynamic_yaml.load(f)
        return cfg
