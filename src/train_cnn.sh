#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:./"
DATA_DIR=$1
METADATA_DIR=$2
TEST_FILE_PATH=$3

TARGET_DIR="${DATA_DIR}/target/"
HEIGHT_PATH="${DATA_DIR}/height.txt"
PCA_MODEL_PATH="${DATA_DIR}/pca_model.jlb"

VIC_KEYPOINT_PATH="${METADATA_DIR}/vic_sparse_key_verts.npy"

echo 'DATA_DIR = ' $DATA_DIR
echo 'TARGET_DIR = ' $TARGET_DIR
echo 'HEIGHT_PATH = ' $HEIGHT_PATH
echo 'PCA_MODEL_PATH = ' $PCA_MODEL_PATH
echo 'VIC_KEYPOINT_PATH = ' $VIC_KEYPOINT_PATH
echo 'TEST_FILE_PATH = ' $TEST_FILE_PATH

if test -f $PCA_MODEL_PATH; then
    echo
else
    echo 'PCA model path does not exist: ' $PCA_MODEL_PATH
    exit
fi

if test -f $VIC_KEYPOINT_PATH; then
    echo
else
    echo 'victoria keypoint file path does not exist: ' $VIC_KEYPOINT_PATH
    exit
fi

echo '\n\nstart training front  model'
python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type f -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH    \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 120 \
        -early_stop_patient 15 \
        -use_pose_variant 1 \
        -encoder_type densenet \
        -mesh_loss_vert_idxs_path $VIC_KEYPOINT_PATH \
	      -is_color \
	      -in_test_file $TEST_FILE_PATH

echo '\n\nstart training side model'
python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type s -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH    \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 120 \
        -early_stop_patient 15 \
        -use_pose_variant 1 \
        -encoder_type densenet \
        -mesh_loss_vert_idxs_path $VIC_KEYPOINT_PATH \
	      -is_color \
	      -in_test_file $TEST_FILE_PATH

echo '\n\nstart training joint model'
python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type joint -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH  \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 120 \
        -early_stop_patient 15 \
        -use_pose_variant 1 \
        -encoder_type densenet \
        -mesh_loss_vert_idxs_path $VIC_KEYPOINT_PATH \
	      -is_color \
	      -in_test_file $TEST_FILE_PATH

#echo '\n\nconvert front pytorch pretrained weight to tensorflow graph'
#IN_MODEL_PATH_F="$DATA_DIR/models/f/final_model.pt"
#OUT_MODEL_PATH_F="$DATA_DIR/models/shape_model_f.jlb"
#
#if test -f "$IN_MODEL_PATH_F"; then
#    echo 'start convert model: ' $IN_MODEL_PATH_F
#    python ./pca/tool_convert_torch_to_tf.py -in_model_path $IN_MODEL_PATH_F -out_model_path $OUT_MODEL_PATH_F  -vic_mesh_path $VIC_MESH_PATH
#else
#    echo 'something wrong. the model path does not exist: ' $IN_MODEL_PATH_F
#fi
