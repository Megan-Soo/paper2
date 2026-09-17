import polyscope as ps
import polyscope.imgui as psim
import numpy as np

"""
    steps = {
                "1. Original lung mask":"Binarises labels<=2. Works either when both lungs are assigned 1 or when lungs are assigned 1 and 2.",
                "2. Erode peripheral lung":"sitk.Ball with radius 3 removes the outer lung voxels",
                "3. Apply white tophat filtering":"Filtering determined by structural element. Finds bright spots and small details smaller than a structuring element by subtracting an image's morphological opening from the original image.",
                "4. Apply enhanced mean filtering":"Filtering determined by neighbourhood size.",
                "5. Combine filtered results":"White tophat + enhanced mean",
                "6. Label propagation":"Dilate a little to try to rejoin small broken connections",
                "7. Remove small unconnected components":f"Components <1000 voxels are removed."
    }
"""

# Load saved arrays
sample_child = 'visualise_pipeline/get_vessel_mask_001.npz'
sample_adult = 'visualise_pipeline/get_vessel_mask_EXAM5332.npz'
path = sample_adult
arrs = np.load(path,allow_pickle=True)
steps = arrs['steps'].item()
step1 = arrs['step1']
step2 = arrs['step2']
step3 = arrs['step3']
step4 = arrs['step4']
step5 = arrs['step5']
step6 = arrs['step6']
step7 = arrs['step7']
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
            
        elif curr_frame==4:
            arr=step4
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("mask", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)
            
        elif curr_frame==5:
            arr=step5
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("mask", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)
            
        elif curr_frame==6:
            arr=step6
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("mask", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)
            
        elif curr_frame==7:
            arr=step7
            indices = np.argwhere(arr)
            grid = ps.register_sparse_volume_grid("mask", origin,spacing,indices)
            grid.add_scalar_quantity("vals",arr[arr!=0],defined_on='cells',enabled=True)

        else:
            print('invalid step')
            pass
                                                                        
ps.init()
ps.set_user_callback(callback)
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([0,0,0,0])
ps.set_navigation_style("free")
ps.set_ground_plane_mode("none")
ps.show()