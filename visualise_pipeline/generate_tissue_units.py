import polyscope as ps
import polyscope.imgui as psim
import numpy as np

"""
steps = {"1a. Labelled lobes mask": "Output of vessel_to_lobe.py",
        "1b. Erode lobes mask by kernel radius 2": f"Leaving behind the 'core' region where conducting airways are likely to occupy.",
        "1c. Remove vessels from lobes mask":"Retain lung parenchyma voxels for calculating normalised lung tissue density.",
        "2a. Raw lung MRI": "",
        "2b. Median-filtered lung MRI": "Apply a median filter with radius 2. [med_filtered.nii.gz].",
        "3. Normalised tissue density":"Median-filtered lung parenchyma signals normalised against the average chest wall muscle signal & corrected for lung & muscle decay times at 3T. [density_mask.nii.gz, density_histogram.png].",
        "4. Density bins":"Divide the density range into 10 bins and assign labels to the respective regions. [density_bins.nii.gz]",
        "5. Generate acini tissue units":"Median density values are used to obtain the relative spatial distribution of acini and their relative volumes (unscaled).",
        "6. Estimate acini volumes":"Linearly scale the acini volumes so that their total sums to the volume of lung parenchyma voxels.",
        "7. Assign acini to lobes":"For each lobe, export [.ipdata] for grow_lobes.py & save acini volumes in [.exdata]."
    }
"""

# region:Load saved arrays
sample_child = 'visualise_pipeline/generate_tissue_units_001.npz'
sample_adult = 'visualise_pipeline/generate_tissue_unitsEXAM5332.npz'
path = sample_child
arrs = np.load(path,allow_pickle=True)
steps = arrs['steps'].item()
spacing = arrs['spacing']
step1a = arrs['step1a']
step1b = arrs['step1b']
step1c = arrs['step1c']
step2a = arrs['step2a']
step2b = arrs['step2b']
step3 = arrs['step3']
step4 = arrs['step4']
step5_1a = arrs['step5_1a']
step5_1b = arrs['step5_1b']
step6_1a = arrs['step6_1a']
step6_1b = arrs['step6_1b']
step6_2a = arrs['step6_2a']
step6_2b = arrs['step6_2b']
step6_3a = arrs['step6_3a']
step6_3b = arrs['step6_3b']
step6_4a = arrs['step6_4a']
step6_4b = arrs['step6_4b']
step6_5a = arrs['step6_5a']
step6_5b = arrs['step6_5b']
# endregion

# region:shift volumes to +z axis
step5_1a[:,2] = step5_1a[:,2]+spacing[2]*step4.shape[2]
step6_1a[:,2]=step6_1a[:,2]+spacing[2]*step4.shape[2]
step6_2a[:,2]=step6_2a[:,2]+spacing[2]*step4.shape[2]
step6_3a[:,2]=step6_3a[:,2]+spacing[2]*step4.shape[2]
step6_4a[:,2]=step6_4a[:,2]+spacing[2]*step4.shape[2]
step6_5a[:,2]=step6_5a[:,2]+spacing[2]*step4.shape[2]
# endregion

vminmax = (np.min([np.min(a) for a in [step6_1b,step6_2b,step6_3b,step6_4b,step6_5b]]),
           np.max([np.max(a) for a in [step6_1b,step6_2b,step6_3b,step6_4b,step6_5b]]))
origin = [0,0,0]
curr_frame=1
n_timestep=6
def callback():
    psim.PushItemWidth(200)
    # print all steps
    for k,v in steps.items():
        psim.TextUnformatted(k)
        psim.TextUnformatted(v)
        psim.Separator()
    psim.PopItemWidth()

    global curr_frame

    update_frame_data = True

    # Slider to manually scrub through frames  
    slider_updated, curr_frame = psim.SliderInt(f"Step {curr_frame}", curr_frame, 1, n_timestep)
    update_frame_data = update_frame_data or slider_updated

    # Update the scene content if-needed
    if update_frame_data:
        ps.remove_all_structures() # clear
        if curr_frame==1:
            arr=step1a
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("1a", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)

            arr=step1b
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("1b", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)

            arr=step1c
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("1c", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)

        elif curr_frame==2:
            arr=step2a
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("2a", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True,cmap='gray')

            arr=step2b
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("2b", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True,cmap='gray')

        elif curr_frame==3:
            arr=step3
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("3", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True,cmap='jet',vminmax=(0,1))
            
        elif curr_frame==4:
            arr=step4
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("4", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True,datatype='categorical')
            
        elif curr_frame==5:
            pc = ps.register_point_cloud("5",step5_1a,radius=0.02)
            pc.add_scalar_quantity("volume",step5_1b,cmap='jet',enabled=True)
            
        elif curr_frame==6:
            pc = ps.register_point_cloud("RUL",step6_1a,radius=0.02)
            pc.add_scalar_quantity("volume",step6_1b,cmap='jet',vminmax=vminmax,enabled=True)
            pc.set_point_radius_quantity("volume")
            pc = ps.register_point_cloud("RML",step6_2a,radius=0.02)
            pc.add_scalar_quantity("volume",step6_2b,cmap='jet',vminmax=vminmax,enabled=True)
            pc.set_point_radius_quantity("volume")
            pc = ps.register_point_cloud("RLL",step6_3a,radius=0.02)
            pc.add_scalar_quantity("volume",step6_3b,cmap='jet',vminmax=vminmax,enabled=True)
            pc.set_point_radius_quantity("volume")
            pc = ps.register_point_cloud("LUL",step6_4a,radius=0.02)
            pc.add_scalar_quantity("volume",step6_4b,cmap='jet',vminmax=vminmax,enabled=True)
            pc.set_point_radius_quantity("volume")
            pc = ps.register_point_cloud("LLL",step6_5a,radius=0.02)
            pc.add_scalar_quantity("volume",step6_5b,cmap='jet',vminmax=vminmax,enabled=True)
            pc.set_point_radius_quantity("volume")
            
        else: # reset to 1
            curr_frame=1
            update_frame_data = True
            pass
                                                                        
ps.init()
ps.set_user_callback(callback)
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([0,0,0,0])
ps.set_navigation_style("free")
ps.set_ground_plane_mode("none")
ps.show()