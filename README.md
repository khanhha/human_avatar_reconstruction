# install requirements
```python
conda env create -f environment.yml
```

# run the pre-trained models
- download [the models](https://drive.google.com/open?id=1l_Tc83U2ZVafjaq6XunPkLrTdrq93RCS) and put them under the folder ./deploy_models
- move to the folder source
- run the code
```python
export PYTHONPATH="${PYTHONPATH}:./"
python ./deploy/hm_pipeline.py -model_dir MODEL_DIR -img_f path_to_front_img -img_s path_to_side_img -height height_in_meter_of_subject gender 0_if_female_else_1 -out_obj_path obj_path_to_export_mesh_prediction
```

# how to train model
- download the dataset from this link
- denote DATA_DIR point to the root directory of the dataset
- run the following commands: this shell script will sequentially train front, side and then the joint model
```python
cd ./src
sh train_cnn.sh DATA_DIR
```

# documentation
## body reconstruction
- [an overview of cnn-based pipeline](./notes/cnn_pipeline.md)
- [improvement ideas for the cnn-based method ](./notes/cnn_improvement_list.md)
- [a summary of the effect of camera properties on silhouette](./notes/cnn_camera_effect.md)
- [testing ideas for the cnn-based method](notes/testing_ideas.md)
- [a summary of the slice-based method](./notes/slice_method_summary.md)
- [victoria-caesar deformatin pipeline](./notes/vic_mpii_deformation_pipeline.md)
## head reconstruction
- [head reconstruction pipeline](./notes/head_reconstruction.md)
- [head reconstruction paper list](./notes/head_reconstruction_paper_list.md)