<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- code_chunk_output -->

- [Notation](#notation)
- [Datasets](#datasets)
- [Overview](#overview)
- [Data Generation](#data-generation)
  - [Overview](#overview-1)
  - [PCA Model Training](#pca-model-training)
  - [Silhouette generation](#silhouette-generation)
  - [Post processing silhouette and prepare CNN training data](#post-processing-silhouette-and-prepare-cnn-training-data)
- [Training process](#training-process)
  - [Overview](#overview-2)
  - [How to train](#how-to-train)
    - [Training](#training)
    - [Training error visualization](#training-error-visualization)
- [CNN architectures](#cnn-architectures)

<!-- /code_chunk_output -->

# Notation
- caesar meshes with maxplank topology: mpii-caesar mesh
- caesar meshes with victoria topology: vic-caesar mesh

# Datasets
- Orginal MPII-Caesar [dataset](https://drive.google.com/open?id=1x1ChF_34GAA88lEMaybsPlghdk3DjKga) in OBJ format

- MPII_Caesar [dataset](https://drive.google.com/open?id=1dxneLcuc3m32EAgW_UjPv539pmX5ymVG) in Victoria's topology
  - This dataset is created by embedding the Maxplank Ceasar meshes to the Victoria's  topology. As far as I know, there are two main reasons for this: First, Oh says that Victoria is our standard topology, which is also used in other tasks as well. Second, we are not allowed to use Maxplank Caesar mesh topology for commercial project.

  - Because Victoria's triangle list is very big (each mesh is around 5mb), and the triangle list is the same for all the meshes, the dataset just stores the vertex array of each mesh. For visualization, please use [this script](https://github.com/khanhha/human_estimation/blob/master/src/pca/vic_vert_to_mesh.py) to convert vertex array to obj file. This script just simplies attaches the vertex array of each mesh to the triangle list of Victoria.

# Overview
The input/output pair of our shape deep learning (dl) model are described below.
- input: front silhouette, side silhouette, gender, height
- output: 51 float values. the fist value is an indicator value for male/female PCA model. The remaining 50 values are PCA parameters to the corresponding male/female PCA model.   

From the input, the shape dl model predicts 51 values, which are then used to pick the corresponding male/female model to reconstruct a Victoria mesh that represents the customer's shape. In the following sections, I will explain in detail the two main stages of the pipeline

- the __data generation__ stage that prepares the training data for training shape dl model.

- the __dl training__ stage that consists of 3 main phases: training the front model, side model and finally, the joint model.

# Data Generation

The training data generation stage consists of 3 main steps
- __maxplank-to-vic embedding__: Embed Maxplank caesar meshes to Victoria topology.

- __pca model training__: train male/female PCA models from the embedded vic-caeasr meshes. This step also does two more following things.

  - __synthesis__ [optional]:  from the trained male/female PCA models, we can also synthesize new meshes by sampling the covariance matrix of PCA models.

  - __PCA transformation__: transform all vic-caesar meshes to PCA parameters. Specifically, each mesh of 72526 vertices will be compressed to 50 PCA parameters. These PCA parameters will be the training target (y values in the pair (x->y)) for training shape dl models.

- __silhouette projection__: in this stage, we load all the vic-caesar meshes into Blender, do pose variants synthesis, and project to front/side silhouettes

In the following sections, I will describe each step in detail and provide instructions to bring them up.

## Overview

&NewLine;

  - __silhouette generation__
    - <img src = 'https://g.gravizo.com/svg?%20digraph%20G%20{%20syn_ml_mesh[label=%22synthesized%20male%20mesh:%2030000x72576x3%22%20shape=box]%20syn_fml_mesh[label=%22synthesized%20female%20mesh:%2030000x72576x3%22%20shape=box]%20bl_ml_sil[label=%22blender:%20male%20sil%20projection%22%20shape=box]%20bl_fml_sil[label=%22blender:%20female%20sil%20projection%22%20shape=box]%20sil_fml_post[label=%22female%20sil%20post-processing%20\n%20binarization,%20height%20normalization%22%20shape=box]%20sil_ml_post[label=%22male%20sil%20post-processing%20\n%20binarization,%20height%20normalization%22%20shape=box]%20final_sil_ds[label=%22final%20sil%20dataset:%20\n%20male:%2030000x2x384x256%20\n%20female:%2030000x2x384x256%22%20shape=box]%20syn_ml_mesh%20-%3E%20bl_ml_sil%20syn_fml_mesh%20-%3E%20bl_fml_sil%20bl_ml_sil%20-%3E%20sil_ml_post%20bl_fml_sil%20-%3E%20sil_fml_post%20sil_ml_post%20-%3E%20final_sil_ds%20sil_fml_post%20-%3E%20final_sil_ds%20}'/>

## PCA Model Training

Instruction to bring up [this step](./cnn_pipeline_instruction.md#PCA-training)

The PCA model we use is a simplified version of [the SMPL model](http://files.is.tue.mpg.de/black/papers/SMPL2015.pdf), which encodes the shape and pose of human space. However, in our model, we ignore the pose parameters because out images just have A pose. Please check the referenced paper for more detail about PCA human model.

Our target in this step is two victoria-based PCA models for male/female, each of which consists of 50 principal components with dimension of 50x72576x3 where 72576 is the number of victoria vertices and 3 represents 3 dimension x,y,z. Intuitively, each principal component of 72675x3 represents vertex deviations from the mean victoria. For example, the first component will encode the variations of human subjects along the "height" direction. Adding the mean mesh with multiplication of the first "height" component will create new subjects of different heights.

The below diagram illustrates the general flow of PCA training, which includes two branches: the left one for male PCA model and the right one for the female PCA model. Let's talk about the left branch; the right one is similar.

The input data to male pca training is 2150 vic-caesar meshes of 72567 (x,y,z) vertices. The output of male pca training are 50 PCA components of 72567 (x,y,z) principal deviations from the mean mesh.

From the trained male PCA model, we can [optional] synthesize, for example, 30000 new meshes of 72576 vertices.

These 2150 original meshes (or 30000 synthesized meshes) is then compressed from the mesh representation of 72576 vertices to the PCA representation of 50 PCA values: 2150x50 or 30000x50.

Our final data has shape of 60.000x51 (50+1), where one additional value indicates which PCA model is used: male or female. If this value is 0, the target belongs to female PCA model; otherwise, it belongs to male PCA model.

&NewLine;
   - <img src='https://g.gravizo.com/svg?%20digraph%20G%20{%20ml_mesh[label=%22male%20meshes:%202150x72576x3%22%20shape=box]%20fml_mesh[label=%22female%20meshes::%202150x72576x3%22%20shape=box]%20ml_pca_train[label=%22male%20pca%20training%22%20shape=box]%20fml_pca_train[label=%22female%20pca%20training%22%20shape=box]%20ml_pca[label=%22male%20PCA%20model:%2050x72576x3%22%20shape=box]%20fml_pca[label=%22female%20PCA%20model:%2050x72576x3%22%20shape=box]%20syn_ml_mesh[label=%22synthesized%20male%20mesh:%2030000x72576x3%22%20shape=box]%20syn_fml_mesh[label=%22synthesized%20male%20mesh:%2030000x72576x3%22%20shape=box]%20syn_ml_pca[label=%22male%20pca%20values:%2030000x50%22%20shape=box]%20syn_fml_pca[label=%22female%20pca%20values:%2030000x50%22%20shape=box]%20final_ds[label=%22final%20pca%20dataset:%2060000x51%20\n%20(+1%20indicator%20for%20male%20or%20female)%22%20shape=box]%20ml_mesh-%3Eml_pca_train%20fml_mesh-%3Efml_pca_train%20ml_pca_train%20-%3E%20ml_pca%20fml_pca_train%20-%3E%20fml_pca%20ml_pca%20-%3E%20syn_ml_mesh%20fml_pca%20-%3E%20syn_fml_mesh%20syn_ml_mesh%20-%3E%20syn_ml_pca%20syn_fml_mesh%20-%3E%20syn_fml_pca%20syn_ml_pca%20-%3E%20final_ds%20syn_fml_pca%20-%3E%20final_ds%20}'/>


## Silhouette generation
Instruction for [this step](./cnn_pipeline_instruction.md#silhouette-generation)

In this step, we use Blender to project original/synthesized meshes to silhouettes. Given a vic-caesar mesh, we also create around 30 pose variants per mesh. Specifically, for front silhouette projection, we rig the mesh by randomly changing the angle between two leg bones or two arm bones. For side silhouette projection, we randomly change spine and neck angles.  

The zoom-in flow of the step is described in below.
- Load a new vertex array that represent a new human subject.
- Estimate all joint locations based on Victoria's topology. For example, the left shoulder joint is estimated as the average of the vertex group [0,20,99,800,250,..], or the right hip joint is the average of the vertex group [90, 600, 6500, 40000,...]
- From the estimated skeleton and the mesh, compute rigging weights using Blender python APIs.
- For front silhouettes, randomly change arm, leg bone angles and project to front silhouettes.
- For side silhouettes, randomly change spine, neck angles and project to side silhouettes.

## Post processing silhouette and prepare CNN training data
the code in this stage performs the following tasks
- transform silhouette projected by Blender to binary image
- do height normalization:
    + rescale all silhouette to the same height
    + center silhouette within the image
- split dataset including silhouettes and PCA values into train, valid, test set

Below are the commands to run the code


# Training process

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

## How to train
### Training
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
### Training error visualization
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
