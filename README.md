
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [install requirements](#install-requirements)
- [run the pre-trained body models](#run-the-pre-trained-body-models)
- [run measurement calculation on predicted vertices](#run-measurement-calculation-on-predicted-vertices)
- [install libigl library for head estimation](#install-libigl-library-for-head-estimation)
- [run head estimation code](#run-head-estimation-code)
  - [download model data](#download-model-data)
  - [download test data](#download-test-data)
  - [run code](#run-code)
- [training](#training)
- [documentation](#documentation)
  - [body reconstruction](#body-reconstruction)
  - [head reconstruction](#head-reconstruction)

<!-- /code_chunk_output -->

# install requirements
## install base packages
```python
conda env create -f environment.yml
```
## install other packges
- install [tensorflow openpose](https://github.com/ildoonet/tf-pose-estimation)
- conda install -c conda-forge shapely 
- pip install triangle

## docker
sudo vim /etc/docker/daemon.json
add the following line
{
    "dns":["192.168.1.1", "8.8.8.8"]
}
sudo systemctl daemon-reload
sudo systemctl restart docker
- remove manual libs from environment.yml: tf-pose, pycoco

# run the pre-trained body models
- download all the files in [the models directory](https://drive.google.com/open?id=1wNr1uVJH27uRHZZLDnLjjQA38HQ4ltSF) and put them under the folder ./deploy_models
- move to the folder source
- download [the test data](https://drive.google.com/open?id=1BLL8VAjId6qBA3p6ebytQRsY7Xk0hCp7)
- run the code
```python
export PYTHONPATH="${PYTHONPATH}:./"
python ./deploy/hm_pipeline.py -model_dir MODEL_DIR/deploy_models -in_txt_file TEST_DATA_DIR/data.txt
```

# run measurement calculation on predicted vertices
- for more information about measurements, please refer to the chart picture in __/notes/woman_measurement_chart.jpg__
- download all the files [in the meta-data diretory](https://drive.google.com/open?id=1eW6eaF8LTYbwmOY-fqrC1Lc08J48BN_W), put them under ./deploy_models/meta_data/
- move to the folder source
- run the code
```python
export PYTHONPATH="${PYTHONPATH}:./"
python ./deploy/hm_measurement.py -obj path_to_obj_file -grp ./deploy_models/meta_data/victoria_measure_vert_groups.pkl -nbr ./deploy_models/meta_data/victoria_measure_contour_circ_neighbor_idxs.pkl
```

# install libigl library for head estimation
You can ignore that part if you don't need to run head estimation.
The library [libigl](https://libigl.github.io) is used in our code to solve the biharmonic problem for reconstructing the head vertices from the boundary
information including facial vertices and neck vertices. Currently, we use the dev branch from the libigl reposritory because,
for some unknown reason, the master branch doesn't work. To maintain consistency, I cloned the dev branch and put it under thirdparty/libigl.
The below installament instructions is copied from [the library's homepage](https://libigl.github.io/example-project/).
The building takes around 5 minutes on my core I7 PC to complete.

```python
conda activate environment_name
cd third_parties/libigl
mkdir build
cd build
cmake ..
make
```
Check the folder **third_parties/libigl/python** to see if **pyigl.so** is built sucessfully.


This library is imported in the code deploy/hm_head_model.py as follows

```python
import sys
sys.path.insert(0, '../../third_parties/libigl/python/')
import pyigl as igl
```

# run head estimation code
## download model data
in addition to body models, the head estimation requires the following models:
- ***dlib_shape_predictor_68_face_landmarks.dat***: dlib model for 68 facial landmarks detection
- ***face_parsing_model.pth***: a pytorch [model](https://github.com/zllrunning/face-parsing.PyTorch) that segment face into eyes, nose, mouth, etc.
- PRN facelib models inside the folder ***MODEL_DIR/prn_facelib_data***

to update data for the code, please download the following things.
- download all files in [the models direcotory](https://drive.google.com/open?id=1wNr1uVJH27uRHZZLDnLjjQA38HQ4ltSF) and put them under the folder MODEL_DIR
- download all the files in [the meta-data directory](https://drive.google.com/open?id=1eW6eaF8LTYbwmOY-fqrC1Lc08J48BN_W) and put them under the folder META_DATA_DIR

## download test data
- download test face imges data from [the link](https://drive.google.com/open?id=16Cgdb4D0IDBI9aEBPW6wBLHkx39Tdwiu) and put all the images under the folder IMG_TEST_FACE_DIR

## run code

please make sure that you already installed the libigl library before running the code

```python
export PYTHONPATH="${PYTHONPATH}:./"

python deploy/hm_pipeline_full_demo.py
-model_dir MODEL_DIR
-meta_data_dir META_DATA_DIR
-in_txt_file IMG_TEST_FACE_DIR/data.txt
-out_dir OUTPUT_DIR
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
