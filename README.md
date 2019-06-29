# install requirements
```python
conda env create -f environment.yml
```

# run the pre-trained models
- download [the models](https://drive.google.com/open?id=1l_Tc83U2ZVafjaq6XunPkLrTdrq93RCS) and put them under the folder ./deploy_models
- download [the latest shape model](https://drive.google.com/open?id=1ue5UYiTWyuc1t2wWOj7X4OqVPdRSI-AR) (trained on the original dataset) and replace the old corresponding ones in the folder ./deploy_models with them
- download [the test data](https://drive.google.com/open?id=1BLL8VAjId6qBA3p6ebytQRsY7Xk0hCp7)
- move to the folder source
- run the code
```python
export PYTHONPATH="${PYTHONPATH}:./"
python ./deploy/hm_pipeline.py -model_dir MODEL_DIR/deploy_models -in_txt_file TEST_DATA_DIR/data.txt
```

# run measurement calculation on predicted vertices
- for more information about measurements, please refer to the chart picture in __/notes/woman_measurement_chart.jpg__
- download the [meta-data](https://drive.google.com/open?id=1YUdRzxdwfDj9-QXz9NScr9SaMWKkabKP), put them under ./deploy_models/meta_data/ 
- move to the folder source
- run the code
```python
export PYTHONPATH="${PYTHONPATH}:./"
python ./deploy/hm_measurement.py -obj path_to_obj_file -grp ./deploy_models/meta_data/victoria_measure_vert_groups.pkl -nbr ./deploy_models/meta_data/victoria_measure_contour_circ_neighbor_idxs.pkl
```

# training
for instruction to generate training data and training steps, please refer to the note ./notes/cnn_pipeline.md

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