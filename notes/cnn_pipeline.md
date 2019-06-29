# tale of content
- [notations](#notation)
- [training data preparation](#training-data-preparation)
- [CNN architectures](#CNN-architectures)
- [training process](#training-process)

# notation
- caesar meshes with maxplank topology: mpii-caesar mesh
- caesar meshes with victoria topology: vic-caesar mesh

# Data Generation

## Overview
- __what is a victoria-based PCA model?__
    - a victoria-based PCA model of 50 principal components has dimension of 50x72576x3 where 72576 is the number of victoria vertices and 3 represents 3 dimension x,y,z
    - each component of 72675x3 represents vertex devications from the mean victoria
- __PCA models training__
    - the PCA model is trained using [IncrementalPCA](https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html#sphx-glr-auto-examples-decomposition-plot-incremental-pca-py) in sklearn.
   The incremental PCA model is chosen because the normal PCA model requires the whole training data to be loaded on memory, which is impossible with a PC of 16GB Ram.

&NewLine;
   - <img src='https://g.gravizo.com/svg?%20digraph%20G%20{%20ml_mesh[label=%22male%20meshes:%202150x72576x3%22%20shape=box]%20fml_mesh[label=%22female%20meshes::%202150x72576x3%22%20shape=box]%20ml_pca_train[label=%22male%20pca%20training%22%20shape=box]%20fml_pca_train[label=%22female%20pca%20training%22%20shape=box]%20ml_pca[label=%22male%20PCA%20model:%2050x72576x3%22%20shape=box]%20fml_pca[label=%22female%20PCA%20model:%2050x72576x3%22%20shape=box]%20syn_ml_mesh[label=%22synthesized%20male%20mesh:%2030000x72576x3%22%20shape=box]%20syn_fml_mesh[label=%22synthesized%20male%20mesh:%2030000x72576x3%22%20shape=box]%20syn_ml_pca[label=%22male%20pca%20values:%2030000x50%22%20shape=box]%20syn_fml_pca[label=%22female%20pca%20values:%2030000x50%22%20shape=box]%20final_ds[label=%22final%20pca%20dataset:%2060000x51%20\n%20(+1%20indicator%20for%20male%20or%20female)%22%20shape=box]%20ml_mesh-%3Eml_pca_train%20fml_mesh-%3Efml_pca_train%20ml_pca_train%20-%3E%20ml_pca%20fml_pca_train%20-%3E%20fml_pca%20ml_pca%20-%3E%20syn_ml_mesh%20fml_pca%20-%3E%20syn_fml_mesh%20syn_ml_mesh%20-%3E%20syn_ml_pca%20syn_fml_mesh%20-%3E%20syn_fml_pca%20syn_ml_pca%20-%3E%20final_ds%20syn_fml_pca%20-%3E%20final_ds%20}'/>


&NewLine;

  - __silhouette generation__
    - <img src = 'https://g.gravizo.com/svg?%20digraph%20G%20{%20syn_ml_mesh[label=%22synthesized%20male%20mesh:%2030000x72576x3%22%20shape=box]%20syn_fml_mesh[label=%22synthesized%20female%20mesh:%2030000x72576x3%22%20shape=box]%20bl_ml_sil[label=%22blender:%20male%20sil%20projection%22%20shape=box]%20bl_fml_sil[label=%22blender:%20female%20sil%20projection%22%20shape=box]%20sil_fml_post[label=%22female%20sil%20post-processing%20\n%20binarization,%20height%20normalization%22%20shape=box]%20sil_ml_post[label=%22male%20sil%20post-processing%20\n%20binarization,%20height%20normalization%22%20shape=box]%20final_sil_ds[label=%22final%20sil%20dataset:%20\n%20male:%2030000x2x384x256%20\n%20female:%2030000x2x384x256%22%20shape=box]%20syn_ml_mesh%20-%3E%20bl_ml_sil%20syn_fml_mesh%20-%3E%20bl_fml_sil%20bl_ml_sil%20-%3E%20sil_ml_post%20bl_fml_sil%20-%3E%20sil_fml_post%20sil_ml_post%20-%3E%20final_sil_ds%20sil_fml_post%20-%3E%20final_sil_ds%20}'/>

## PCA Model Training
The PCA model we use is a simplified version of [the SMPL model](http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf), which encodes the shape and pose of human space. However, in our model, we ignore the pose parameters because out images just have A pose.  In our project, there are two separated PCA models for male and female. The male model is trained from around 2000 male meshes  and the female model is trained from the female meshes in the Caesar dataset. To train the models, run the following code

download [the dataset](https://drive.google.com/open?id=1dxneLcuc3m32EAgW_UjPv539pmX5ymVG)
```python
export PYTHONPATH="${PYTHONPATH}:./"
python ./pca/pca_vic_train.py -vert_dir CAESAR_VERT_DIR -vic_mesh_path VIC_OBJ_PATH -female_names_file TXT_FEMALE_NAMES_PATH -out_dir OUTPUT_DIR -n_synthesize_samples 100
```

For further information about the parameters, please refer to the help from the code. In general, the code performs the following tasks
- train two pca models for male and femle
- export the first 10 principal components for each PCA model for the sake of debugging.
- synthesize more PCA values for each model, if the param __n_synthesize_samples__ is greater than zero
- export a number of OBJ debug meshes for PCA values.

## Silhouette Generation
After the origin meshes are transformed to PCA space and new PCA values are synthesized, we need to find the corresponding front/site silhouettes for PCA values. This is done by a Blender script which loads vertex arrays of caesar meshes generated from the previous step and project them to front/side silhouette.

Do the following steps to generate silhouettes.
 - __notice__: it will take very long to generate silhouettes for all meshes; therefore, you should modify the script to test a few meshes first.
 - install Blender 2.79b. Other versions are not tested yet.
 - start [the blender file](https://drive.google.com/open?id=1zUyAl8Jz21NT5r3yHKhRQcC5XuU_jpL9): caesar_project_silhouette.blend
 - press NUMPAD key 0 to make sure that the 3D view is in projection camera mode. This step must be done; otherwise, the generated silhouettes will be rendered from the view manipulation camera matrix of Blender.
 - generate male silhouettes
    - update the path variables in the script file (sorry for this inconveniece. Run the script from console with arguments doesn't work)
        - pca_co_dir: points to the path: PCA_MOEL_PATH/verts/male
        - sil_root_dir: points to output male silhouette: TRAINIG_DATA_DIR/male_sil_raw
    - press "Run Script".

 - generate female silhouttes
    - update the path variables:
        - pca_co_dir: points to the path: PCA_MOEL_PATH/verts/female
        - sil_root_dir: points to output female silhouette: TRAINIG_DATA_DIR/female_sil_raw
    - press "Run Script"

## Post processing silhouette and prepare CNN training data
the code in this stage performs the following tasks
- transform silhouette projected by Blender to binary image
- do height normalization:
    + rescale all silhouette to the same height
    + center silhouette within the image
- split dataset including silhouettes and PCA values into train, valid, test set

Below are the commands to run the code

```python
export PYTHONPATH="${PYTHONPATH}:./"
python ./pca/tool_prepare_train_data_ml_fml.py
-sil_f_fml_dir SILOUHETTE_FEMALE_DIR/sil_f_raw/
-sil_s_fml_dir SILOUHETTE_FEMALE_DIR/sil_s_raw/
-sil_f_ml_dir SILOUHETTE_MALE_DIR/sil_f_raw/
-sil_s_ml_dir SILOUHETTE_MALE_DIR/sil_s_raw/
-target_ml_dir PCA_MODEL_DIR/pca_coords/male/
-target_fml_dir PCA_MODEL_DIR/pca_coords/female/
-pca_ml_model_path PCA_MODEL_DIR/vic_male_pca_model.jlb
-pca_fml_model_path PCA_MODEL_DIR/vic_female_pca_model.jlb
-resize_size 384x256
-out_dir OUPUT_CNN_DATA_DIR
```

# training process

## Overview
Training consists of two main steps. First we train the front and side CNN models and then we train the joint model with the pre-trained weights from the front and side models. Training front and side model separately is required because it would help create better than models that learn distinctive features in front and side silhouettes. If we group all the training into one step, the model will bias toward the front silhouette while ignore the side silhouettes.

- train front model
  - input:
    - Nx384x256: N front silhouettes of size 384x256
    - Nx2: (height + gender) pairs
  - target:
    - Nx51 values: 1 gender indicator + 50 pca values

- train side model
  - input:
    - Nx384x256: N side silhouettes of size 384x256
    - Nx2: (height + gender) pairs
  - target:
    - Nx51: 1 gender indicator + 50 pca values

- train joint model based on pre-trained weights from front and side models
  - input:
    - Nx2x384x256: N front and size silhouettes of size 384x256
    - Nx2: (height+ gender) pairs
  - target:
    - NX51: 1 gender indicator + 50 pca values

## how to train
### training
- download the original dataset (4301 meshes) from [this link](https://drive.google.com/open?id=1c9eHv9NBo4PkfpRCHWix1wzCKumsICG3),
or the synthesized dataset (62000) from [this link](https://drive.google.com/open?id=18Kaj8A18wEMiZmmi7y9k9QDmSsFrcQO_)
- denote DATA_DIR point to the root directory of the dataset. For example, ```root_dir/sil_384_256_ml_fml_nosyn```
- run the following commands: this shell script will sequentially train front, side and then the joint model. The final joint model
will be converted from Pytorch to Tensorflow graph and wrapped with additional information for inference.
    ```python
    cd ./src
    sh train_cnn.sh DATA_DIR
    ```
- for more stable training,you can comment the code in the sn_train_cnn.sh to train modes one by one
### training error visualization
- run tensorboard
    ```python
    cd DATASET_DIR/log
    tensorboard ./f #for the front model
    tensorboard ./s #for the side model
    tensorboard ./joint #for the side model
    ```
- open the web brower to check the error
![traing_error](notes/images/training_error.jpg)

- run inference on the model: copy the shape_model.jlb file to the deploy model directory and go back to the "run the pre-trained models" step


# CNN architectures
There are two types of architectures: front/side architecture and joint architecture. The front/side architecture just takes in front or side images with auxiliary inputs (height, gender) and the joint architecture takes in both front and side images. The joint architecture reuses the trained weights from the front/side architectures to extract features from the front and side silhouettes.  This separation is needed is just for the purpose of training the joint model. If we train the joint model from scratch, there is a high change that the model will bias toward the front silhouette while neglecting the side silhouettes. Therefore, we have to train the front/side mode independently before combining their weights in the joint model.
&NewLine;
- Front/Side Architecture

&NewLine;
    <img src='https://g.gravizo.com/svg?%20digraph%20G%20{%20main[label=%22front/side%20silhouette%22%20shape=box];%20cnn[label=%22CNN:%20Densenet%22%20shape=box];%20fcn[label=%22FCN_cnn:%20fully%20connected%20layers%22%20shape=box];%20fcn_1[label=%22FCN_aux:%20fully%20connected%20layers%22%20shape=box];%20fcn_2[label=%22FCN_final:%20fully%20connected%20layers%22%20shape=box];%20aux[label=%22aux_input:%20gender,%20height%22%20shape=box]%20concat[label=%22concatenation%22%20shape=box]%20pca[label=%22output:%201%20gender%20indicator%20+%2050%20pca%20values%22%20shape=box]%20main%20-%3E%20cnn;%20cnn%20-%3E%20fcn;%20aux%20-%3E%20fcn_1;%20fcn%20-%3E%20concat;%20fcn_1%20-%3E%20concat;%20concat%20-%3E%20fcn_2;%20fcn_2%20-%3E%20pca%20}'/>


- Joint Architecture
&NewLine;

  &nbsp;
  <img src='https://g.gravizo.com/svg?%20digraph%20G%20{%20f_sil[label=%22front%20silhouette%22%20shape=box];%20s_sil[label=%22side%20silhouette%22%20shape=box];%20f_cnn[label=%22pre-trained%20front%20CNN%22%20shape=box];%20s_cnn[label=%22pre-trained%20side%20CNN%22%20shape=box];%20fs_concat[label=%22concatenation%22%20shape=box];%20fs_fcn[label=%22fcn:%20fully%20connected%20layers%22%20shape=box];%20aux[label=%22aux_input:%20height,%20gender%22%20shape=box];%20f_fcn_aux[label=%22fcn_aux_f:%20pre-trained%20front%20fcn%20layers%22%20shape=box];%20s_fcn_aux[label=%22fcn_aux_s:%20pre-trained%20side%20fcn%20layers%22%20shape=box];%20aux_elmwise_max[label=%22element-wise%20maximum%22%20shape=box];%20concat_1[label=%22concatenation%22%20shape=box];%20final_fcn[label=%22fcn_final:%20fully%20connected%20layers%22%20shape=box]%20output[label=%22output:%201%20gender%20indicator%20+%2050%20pca%20values%22%20shape=box]%20f_sil-%3Ef_cnn;%20s_sil-%3Es_cnn;%20f_cnn-%3Efs_concat;%20s_cnn-%3Efs_concat;%20fs_concat-%3Efs_fcn;%20aux%20-%3E%20f_fcn_aux;%20aux%20-%3E%20s_fcn_aux;%20f_fcn_aux%20-%3E%20aux_elmwise_max;%20s_fcn_aux%20-%3E%20aux_elmwise_max;%20fs_fcn%20-%3E%20concat_1;%20aux_elmwise_max%20-%3E%20concat_1;%20concat_1%20-%3E%20final_fcn;%20final_fcn%20-%3E%20output%20}'/>
 &NewLine;
