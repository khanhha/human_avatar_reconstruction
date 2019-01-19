python ./src/ffdt_deformation_parameterize_tool.py  -ctl ./data/test_deform/sphere_ctl.obj -tpl ./data/test_deform/sphere_tpl.obj -g 1 -o ./data/test_deform/sphere_param.pkl

python ./src/fftdt_deformation_reconstruct_tool.py -t ./data/test_deform/sphere_tpl.obj -d ./data/test_deform/sphere_df_ctl.obj -p ./data/test_deform/sphere_param.pkl -o ./data/test_deform/sphere_deformed_output.obj
