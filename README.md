# human_estimation
set up enviroment
    conda create --name human_estimation
	conda install -n human_estimation -c conda-forge opencv 
	conda install -n human_estimation scipy numpy matplotlib
	conda activate human_estimation

grun shape_key_analyze.py
    this code preprocess silhouette projections from Blender
    and compare silhouettes of each shape-key at values of 0 and 1
    
instruction to analysize silhouete from blender's silhouette projections
    run blender file to project 3D models to front and side silhouettes
    update directories in the file shape_key_analyze.py to
    run shape_key_analyze.py
