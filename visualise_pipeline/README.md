Pipeline steps:
1. get_vessel_mask.py
2. vessel_to_lobe.py (prereq: get_vessel_mask.py)
3. generate_tissue_units.py
4. get_centreline.py (prereq: vessel_to_lobe, generate_tissue_units)
5. grow_lobes.py (prereq: generate_tissue_units, get_centreline)
