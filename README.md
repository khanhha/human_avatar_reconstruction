# human_estimation
1. set up enviroment \n
	conda create --name human_estimation \n
	conda install -n human_estimation -c conda-forge opencv 
	conda install -n human_estimation scipy numpy matplotlib
	conda activate human_estimation

2. what does shape_key_analyze do?
    this code preprocesses silhouette projections from Blender
    and compare silhouettes of each shape-key at values of 0 and 1
    
3. instruction to analysize silhouete from blender's silhouette projections
    run blender file to project 3D models to front and side silhouettes
    update directories in the file shape_key_analyze.py to
    run shape_key_analyze.py
