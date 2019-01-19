#  How to run deformation algorithm
    1. Install libraries
        conda install numpy scipy

    2.How to run all in one
        1. run on a simple test case of sphere mesh
            cd to human_estimation folder
            sh df_sphere_run.sh

        2. run on human test case
            cd to human_estimation folder
            sh df_body_run.sh

    3.How to run step by step
        - cd to the human_estimation folder

        - calculate parameterization of the template mesh (victoria) with respect to the original control mesh
            python ./src/ffdt_deformation_parameterize_tool.py  -ctl ./data/meta_data/origin_control_mesh_tri.obj -tpl ./data/meta_data/origin_template_mesh.obj -g 1 -o ./data/meta_data/global_parameterization.pkl

        - reconstruct a new template mesh based on the deformed control mesh
            python ./src/fftdt_deformation_reconstruct_tool.py -t ./data/meta_data/origin_template_mesh.obj -d ./data/ctr_mesh/IMG_1928_front_ctl_tri.obj -p ./data/meta_data/global_parameterization.pkl -o ./data/meta_data/output_deformed_mesh.obj

# How to train slice models
    1. install libraries
        conda install numpy scipy matplotlib scikit-learn
        conda install -c conda-forge shapely 
        conda install -c anaconda tensorflow-gpu keras
        pip install dtreeviz
    2. download dataset
    3. run slice_regressor_training_tool.py
# How to run shape-key analysis
1. set up enviroment 
	conda create --name human_estimation
	conda install -n human_estimation -c conda-forge opencv 
	conda install -n human_estimation scipy numpy matplotlib
	conda activate human_estimation

2. what does shape_key_analyze do?
    this code preprocesses silhouette projections from Blender
    and compare silhouettes of each shape-key at values of 0 and 1
    
3. instruction to analysize silhouete from blender's silhouette projections	
	- download blender projects: victoria_silhouette_projection.zip
    	- run two blender projects to project 3D models to front and side silhouettes (update the output directories)
    	- update directories in the file shape_key_analyze.py to point to blender silhouette directories
    	- run shape_key_analyze.py
