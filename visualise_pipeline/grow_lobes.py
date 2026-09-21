import polyscope as ps
import polyscope.imgui as psim
import numpy as np

"""
steps= {"1. Upper airway centreline":"From get_centreline.py",
        "2. Grown tree":"Grown from upper_airway.ipnode/elem/fiel",
        "3. Acini units":"Spatial and volume distribution derived from generate_tissue_units.py. \
            Note that the tree's terminal branches don't map exactly to the generated acini units.",
        "4. Final airway model":"Map volume distribution from generate_tissue_units.py to terminal branches."
        }
"""

# region: Load saved arrays
sample_child = 'visualise_pipeline/grow_lobes_001.npz'
sample_adult = 'visualise_pipeline/grow_lobes_EXAM5332.npz'
path = sample_child
arrs = np.load(path,allow_pickle=True)
steps = arrs['steps'].item()
step1_1a = arrs['step1_1a']
step1_1b = arrs['step1_1b']
step1_1c = arrs['step1_1c']
step2_1a = arrs['step2_1a']
step2_1b = arrs['step2_1b']
step2_1c = arrs['step2_1c']
step3 = arrs['step3']
step4_1a = arrs['step4_1a']
step4_1b = arrs['step4_1b']
# endregion

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
            ps.get_curve_network("upper airway").set_enabled(True)
            ps.get_curve_network("grown").set_enabled(False)
            ps.get_point_cloud("acini").set_enabled(False)
            
        elif curr_frame==2:
            ps.get_curve_network("upper airway").set_enabled(False)
            ps.get_curve_network("grown").set_enabled(True)
            ps.get_point_cloud("acini").set_enabled(False)

        elif curr_frame==3:
            pc=ps.get_point_cloud("acini")
            pc.set_enabled(True)
            pc.update_point_positions(step3)
            pc.set_radius(0.001)
            pc.clear_point_radius_quantity()
            
        elif curr_frame==4:
            pc=ps.get_point_cloud("acini")
            pc.set_enabled(True)
            pc.update_point_positions(step4_1a)
            # pc.set_radius(0.02)
            # pc.set_point_radius_quantity("volume")

        else:
            curr_frame=1
            update_frame_data = True
            pass
                                                                        
ps.init()
# region: initialise all structures
cn=ps.register_curve_network("upper airway",step1_1a,step1_1b,enabled=False)
cn.add_scalar_quantity("radius",step1_1c,defined_on='edges',enabled=False)
cn.set_edge_radius_quantity("radius",False)
cn=ps.register_curve_network("grown",step2_1a,step2_1b,enabled=False)
cn.add_scalar_quantity("radius",step2_1c,defined_on='nodes',enabled=False)
cn.set_node_radius_quantity("radius",False)
pc=ps.register_point_cloud("acini",step3,radius=0.001,color=[255,255,255],enabled=False)
pc.add_scalar_quantity("volume",step4_1b,enabled=True,cmap="jet")
# endregion
ps.set_user_callback(callback)
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([0,0,0,0])
ps.set_navigation_style("free")
ps.set_ground_plane_mode("none")
ps.show()
