# install requirements
```python
conda env create -f environment.yml
```

# download the pre-trained models
- download the shape model and silhouette model and put them under the folder ./models
- move to the folder source
- run the code
```python
python ./deploy/hm_pipeline.py -model_dir MODEL_DIR -img_f path_to_front_img -img_s path_to_side_img -height height_in_meter_of_subject gender 0_if_female_else_1
```
# documentation
- [an overview of cnn-based pipeline](./notes/cnn_pipeline.md)
- [improvement ideas for the cnn-based method ](./notes/cnn_improvement_list.md)
- [a summary of the effect of camera properties on silhouette](./notes/cnn_camera_effect.md)
- [testing ideas for the cnn-based method](notes/testing_ideas.md)
- [a summary of the slice-based method](./notes/slice_method_summary.md)