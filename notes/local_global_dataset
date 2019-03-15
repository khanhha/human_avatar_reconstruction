# data format

The folder slice_local_global_train_data_09_03_2019/data contains data for 23 slice types: Armscye.pkl, Aux_Armscye_Shoulder_0.pkl, etc.

Each slice data file (for eample Armscye.pkl) has following data fields:

- X: NX6 where N is the number of slices. 6 means 3 local ratios plust 3 global ratios (Hip, Waist, Bust). For more detail about the input definition of each slice mode, please refer to to the function SliceModelInputDef._init_local_inputs_mode in the file slice_def.py

- Y: NxK where N is the number of slices. K is the fourier code size. K could be different number depending on each type of slice

For more detail about how the slice data is exported, please refer to the file tool_export_training_data.py

# reconstruct slice points from fourier code
- please refer to the function util.reconstruct_contour_fourier in the file util.py. For more example about this function, please refer to the file tool_slice_regressor_2_test.py









