#!/usr/bin/env python
import os, dynamic_yaml, sys
import numpy as np
import polyscope as ps
from aether.diagnostics import set_diagnostics_on
from aether.indices import ventilation_indices, get_ne_radius
from aether.geometry import define_node_geometry, define_1d_elements, define_rad_from_file, append_units
from aether.geometry import define_init_volume
from aether.ventilation import evaluate_vent
from aether.ventilation import read_params_evaluate_flow
from aether.exports import export_1d_elem_field, export_terminal_solution

def load_config(config_path):
# Load the configuration file using PyYAML
    with open(config_path, 'r') as f:
        cfg = dynamic_yaml.load(f)
    return cfg

def main(subject,weight,ie_ratio):
    cfg = load_config('simulate_ventilation/ventilation.yaml')
    grown_filename = cfg.grown_filename
    if cfg.gradwarp:
        img_analysis_dir = os.path.join(cfg.dirs.img_analysis,subject,'gradwarp')
    else:
        img_analysis_dir = os.path.join(cfg.dirs.img_analysis,subject)
    
    # Set up directories
    if cfg.gradwarp:
        output_directory = os.path.join(cfg.dirs.output_dir,subject,'gradwarp')
    else:
        output_directory = os.path.join(cfg.dirs.output_dir,subject)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    set_diagnostics_on(False)

    # Read settings
    ventilation_indices()
    
    # Read params_evaluate_flow
    print(f' Subject {subject}')
    if not cfg.volume_target:
        cfg.volume_target = weight * 6 * 10**6 # Assume 6 mL/kg. convert L to mm3.
    else:
        volume_target = cfg.volume_target*10**6 # convert L to mm3
    T_interval = cfg.breath_duration #60/18 # respiratory rate of 0.3 Hz == 18 bpm -> 60 sec/18 bpm = 3.33 s
    n_samples = 10-1 # 10 timesteps - 1 FRC @ t=0. t export terminal at sampling intervals of T_interval. To match MoCoLoR timestep, T_sample = T_interval/n_phases
    if cfg.export_terminal_ph is not None:
        path_sample = output_directory+'/terminal' # if '', will not export sample solution
    else:
        path_sample = ''
    press_in = 0.0
    refvol = 0.5
    pmus_step = -196.133
    chest_wall_compliance = cfg.chest_wall_compliance * (10**3/98.0665) # convert mL/cmH2O to mm3/Pa
    print(f" Estimated Chest wall compliance: {cfg.chest_wall_compliance} mL/cmH2O")
    read_params_evaluate_flow(T_interval, press_in, ie_ratio, refvol, volume_target, pmus_step, chest_wall_compliance,n_samples,path_sample)

    define_node_geometry(os.path.join(img_analysis_dir, grown_filename))
    define_1d_elements(os.path.join(img_analysis_dir, grown_filename))
    define_rad_from_file(os.path.join(img_analysis_dir, grown_filename + '_radius'))

    append_units()
    
    # assign initial volume from image
    define_init_volume(os.path.join(img_analysis_dir,cfg.filenames.init_vol_ipfiel))

    # Set the working directory to the this files directory and then reset after running simulation.
    file_location = os.path.dirname(os.path.abspath(__file__))
    cur_dir = os.getcwd()
    os.chdir(file_location)

    # Run simulation.
    evaluate_vent() #needs Parameters folder

    # Set the working directory back to it's original location.
    os.chdir(cur_dir)

    if cfg.export_terminal_ph is not None:
        # Output results
        # Export airway nodes and elements
        # filename = 'airway'
        group_name = 'vent_model'
        # export_1d_elem_geometry(os.path.join(output_directory, filename), group_name)
        # export_node_geometry(os.path.join(output_directory, filename), group_name)

        # Export element field for radius
        filename = 'ventilation_field'
        field_name = 'flow'
        export_1d_elem_field(6, os.path.join(output_directory, filename), group_name, field_name)

        # Export element field for radius
        # filename = 'ventilation_radius_field'
        # ne_radius = get_ne_radius()
        # field_name = 'radius'
        # export_1d_elem_field(ne_radius, os.path.join(output_directory, filename), group_name, field_name)

        # Export terminal solution
        filename = 'term_average'
        export_terminal_solution(os.path.join(output_directory, filename), group_name)
        # export_dvdt(os.path.join(output_directory,filename+'dvdt.txt'),group_name)
        #export_dpdt(os.path.join(output_directory,filename+'dpdt.txt'),group_name)
        # export_vol_press(os.path.join(output_directory,'vol_press.txt'),group_name)

if __name__ == '__main__':
    subject = sys.argv[1]
    weight = float(sys.argv[2])
    ie_ratio = float(sys.argv[3])
    main(subject,weight,ie_ratio)
