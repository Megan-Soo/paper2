#!/usr/bin/env python
import os, sys, argparse
import numpy as np
import polyscope as ps
from aether.diagnostics import set_diagnostics_on
from aether.indices import ventilation_indices, get_ne_radius
from aether.geometry import define_node_geometry, define_1d_elements, define_rad_from_file, append_units
from aether.geometry import define_init_volume
from aether.ventilation import evaluate_vent
from aether.ventilation import read_params_evaluate_flow
from aether.exports import export_1d_elem_field, export_terminal_solution

def main():
    parser = argparse.ArgumentParser(description="Simulate ventilation using init_vol.ipfiel.")
    parser.add_argument("-s", required=False, help="(Optional) Print subject name in terminal.")
    parser.add_argument("--in-dir", required=True, help="Directory containing grown.ipnode/elem/fiel & init_vol.ipfiel files.")
    parser.add_argument("--out-dir",required=True,help="Directory to export files.")
    parser.add_argument("-w","--weight",type=float,help="Subject weight (kg) to estimate chest wall compliance (CWC).")
    parser.add_argument("-v",type=float,help="Subject tidal volume (L).")
    parser.add_argument("-t",type=float,required=True,help="Duration of one breath cycle (s).")
    parser.add_argument("--child",action="store_true",help="CWC adult 3 cmH2O/mL/kg; CWC child 4 cmH2O/mL/kg [Sharp1970]")
    parser.add_argument("-n",type=int,default=0,help="Exports files at n intervals across the breath cycle.")
    parser.add_argument("-ie",type=float,default=1.0,help="Inspiration:Expiration duration ratio.")
    args=parser.parse_args()
    subject = args.subject
    in_dir = args.in_dir
    weight = args.w
    volume_target = args.v
    T_interval = args.t
    child = args.child
    n_samples = args.n
    ie_ratio = args.ie
    output_directory = args.out_dir
    os.makedirs(output_directory,exist_ok=True)
    
    set_diagnostics_on(False)

    # Read settings
    ventilation_indices()
    
    # Read params_evaluate_flow
    print(f' Subject {subject}')
    if not volume_target:
        volume_target = weight * 6 * 10**6 # Assume 6 mL/kg. convert L to mm3.
    n_samples = 10-1 # 10 timesteps - 1 FRC @ t=0. t export terminal at sampling intervals of T_interval. To match MoCoLoR timestep, T_sample = T_interval/n_phases
    if n_samples is not None:
        path_sample = output_directory+'/terminal' # if '', will not export sample solution
    else:
        path_sample = ''
    press_in = 0.0
    refvol = 0.5
    pmus_step = -196.133
    if child: x=4
    else: x=3 # adult
    chest_wall_compliance = weight * x
    print(f" Estimated Chest wall compliance: {chest_wall_compliance} mL/cmH2O")
    chest_wall_compliance = chest_wall_compliance * (10**3/98.0665) # convert mL/cmH2O to mm3/Pa
    read_params_evaluate_flow(T_interval, press_in, ie_ratio, refvol, volume_target, pmus_step, chest_wall_compliance,n_samples,path_sample)

    define_node_geometry(os.path.join(in_dir, "grown"))
    define_1d_elements(os.path.join(in_dir, "grown"))
    define_rad_from_file(os.path.join(in_dir, "grown_radius"))

    append_units()
    
    # assign initial volume from image
    define_init_volume(os.path.join(in_dir, "init_vol.ipfiel"))

    # Set the working directory to the this files directory and then reset after running simulation.
    file_location = os.path.dirname(os.path.abspath(__file__))
    cur_dir = os.getcwd()
    os.chdir(file_location)

    # Run simulation.
    evaluate_vent() #needs Parameters folder

    # Set the working directory back to it's original location.
    os.chdir(cur_dir)

    if output_directory:
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

        
if __name__ == '__main__':
    main()
