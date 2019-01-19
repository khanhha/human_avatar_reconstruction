python ./src/ffdt_deformation_parameterize_tool.py  -ctl ./data/debug/deform/ctl.obj -tpl ./data/debug/deform/tpl.obj -g 1 -o ./data/debug/deform/param.pkl

python ./src/fftdt_deformation_reconstruct_tool.py -t ./data/debug/deform/tpl.obj -d ./data/debug/deform/df_ctl.obj -p ./data/debug/deform/param.pkl -o ./data/debug/deform/output_deformed_mesh.obj
