# data format
    The folder slice_local_global_train_data_09_03_2019/data contains data for 23 slice types: Armscye.pkl, Aux_Armscye_Shoulder_0.pkl, etc.
    
    Each slice data file (for eample Armscye.pkl) has following data fields:
    
    - X: NX6 where N is the number of slices. 6 means 3 local ratios plust 3 global ratios (Hip, Waist, Bust). For more detail about the input definition of each slice mode, please refer to to the function SliceModelInputDef._init_local_inputs_mode in the file slice_def.py
    
    - Y: NxK where N is the number of slices. K is the fourier code size. K could be different number depending on each type of slice

# how to convert the caesar slice data into local_global training format, run the following tool
    tool_export_training_data.py
    -slc_dir SLICE_DATA_DIR/female_slice/
    -feature_dir SLICE_DATA_DIR/slice_code/fourier/ 
    -bad_slc_dir SLICE_DATA_DIR/bad_slices/
    -mode local_global
    -out_dir OUTPUT_DIR
    
    mode specifies different input model definition. it could take the following strings ["single", "local", "local_global", "torso"]
    for more detail about how each input type is defined, please refer to the class SliceModelInputDef in the file slice_def.py 

# reconstruct slice 2D points from its fourier code
    please refer to the function util.reconstruct_contour_fourier in the file util.py. For more example about this function, please refer to the file tool_slice_regressor_2_test.py









