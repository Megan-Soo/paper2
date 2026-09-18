import polyscope as ps
import polyscope.imgui as psim
import numpy as np

"""
steps={"1. Labelled vessel mask":"Output of get_vessel_mask.py is manually labelled in ITKSnap.",
       "2. Get seeds for space partitioning":"Obtained by morphological closing of labelled vessels",
       "3. Resultant lobe mask":"Based on method by Khiati 2024, HAL Open Science."
       }
"""

# Load saved arrays
sample_child = 'visualise_pipeline/vessel_to_lobe_001.npz'
sample_adult = 'visualise_pipeline/vessel_to_lobe_EXAM5332.npz'
path = sample_child
arrs = np.load(path,allow_pickle=True)
steps = arrs['steps'].item()
step1 = arrs['step1']
step2 = arrs['step2']
step3 = arrs['step3']
spacing = arrs['spacing']

origin = [0,0,0]
curr_frame=1
n_timestep=len(steps)
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
        # ps.remove_all_structures() # clear
        if curr_frame==1:
            arr=step1
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("mask", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)

        elif curr_frame==2:
            arr=step2
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("mask", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)
            
        elif curr_frame==3:
            arr=step3
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("mask", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)

        else:
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
