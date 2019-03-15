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

# Pipeline
  ![Alt text](./data/diagrams/slice_extraction.png)
  
  ![Alt text](./data/diagrams/slice_training_model.png)
  
  ![Alt text](./data/diagrams/slice_prediction.png)
  
# How to train slice models
    1. install libraries
        conda install numpy scipy matplotlib scikit-learn
        conda install -c conda-forge shapely 
        conda install -c anaconda tensorflow-gpu keras
        pip install dtreeviz
        
    2. download dataset, denote the path to the dataset is DATA_SET_DIR
	- DATA_SET_DIR/female_slice contains slice contours of caesar females. Each contour consist of (x,y) coordinates. These contours are just for the purpose of visualization. 
	- DATA_SET_DIR/slice_code/fourier contains the actual training data. It consists of width, depth and fourier code of each contour. Note that fourier code is already scaled and normalized. To understand more about the data structure, please check the function load_slice_data in file slice_regressor_training_tool.py
	- DATA_SET_DIR/bad_slices contrains a text file which list bad, irregular file names for each type of slice
	
    3. run tool_slice_regressor_3_train.py 
        -slc_dir DATA_SET_DIR/female_slice/ 
        -feature_dir DATA_SET_DIR/slice_code/fourier/ 
        -bad_slc_dir DATA_SET_DIR/bad_slices/ 
        -model_dir OUPUT_MODEL_DIR
        -slc_ids Hip
	-mode local_global
	
	the parameter bad_slc_dir refer to a folder that contains text files, each of which store bad slice names for each type of slices. Bad slices are ones which are distorted or noise.
	
	the parameter "ids" tell the tool which slice type it should train. The slice id is corresponding to sub folder names under DATA_SET_DIR/female_slice
	
	the parameter "mode" specifies the input information of the model. It  could take the following strings ["single", "local", "local_global", "global", "torso"]. For more detail how each type of input is defined, please refer to the class "SliceModelInputDef" in the file "slice_def.py"
    
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
