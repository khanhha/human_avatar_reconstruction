#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:./"
DATA_DIR=$1

if [ $2 == "" ]
then
  POSE_VARIANT=0
else
  POSE_VARIANT=$2
fi

TARGET_DIR="${DATA_DIR}/target/"
HEIGHT_PATH="${DATA_DIR}/height.txt"
PCA_MODEL_PATH="${DATA_DIR}/pca_model.jlb"

echo 'DATA_DIR = ' $DATA_DIR
echo 'TARGET_DIR = ' $TARGET_DIR
echo 'HEIGHT_PATH = ' $HEIGHT_PATH
echo 'PCA_MODEL_PATH = ' $PCA_MODEL_PATH
echo 'Training with N pose variant = ' $POSE_VARIANT

if test -f $PCA_MODEL_PATH; then
    echo
else
    echo 'PCA model path does not exist: ' $PCA_MODEL_PATH
    exit
fi
echo '\n\nstart training side model'
python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type s -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH    \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 120 \
        -early_stop_patient 15 \
        -n_pose_variant $POSE_VARIANT \
        -encoder_type densenet \
        -mesh_loss_vert_idxs_path /media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/vic_sparse_key_verts.npy \
        -is_color
