#!/usr/bin/env bash
python ./ffdt_param_check.py -p ../data/meta_data/global_parameterization.pkl -ctl ../data/meta_data/origin_control_mesh_tri.obj -tpl ../data/meta_data/origin_template_mesh.obj -ndir ../data/meta_data/neighbor_debug/ -pnt_mesh
../data/meta_data/pointer.obj