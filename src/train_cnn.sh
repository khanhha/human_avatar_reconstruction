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

echo '\n\nstart training front model'
python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type f -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH    \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 30 \
        -n_pose_variant $POSE_VARIANT

echo '\n\nstart training side model'
python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type s -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH    \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 30 \
        -n_pose_variant $POSE_VARIANT

echo '\n\nstart training joint model'
python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type joint -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH    \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 30 \
        -n_pose_variant $POSE_VARIANT

echo '\n\nconvert pytorch pretrained weight to tensorflow graph'
IN_MODEL_PATH="$DATA_DIR/models/joint/final_model.pt"
OUT_MODEL_PATH="$DATA_DIR/models/shape_model.jlb"
VIC_MESH_PATH="$DATA_DIR/victoria_caesar_template.obj"

if test -f "$IN_MODEL_PATH"; then
    echo 'start convert model: ' $IN_MODEL_PATH
    python ./pca/tool_convert_torch_to_tf.py -in_model_path $IN_MODEL_PATH -out_model_path $OUT_MODEL_PATH  -vic_mesh_path $VIC_MESH_PATH
else
    echo 'something wrong. the model path does not exist: ' $IN_MODEL_PATH
fi


echo '\n\nconvert side pytorch pretrained weight to tensorflow graph'
IN_MODEL_PATH_S="$DATA_DIR/models/s/final_model.pt"
OUT_MODEL_PATH_S="$DATA_DIR/models/shape_model_s.jlb"

if test -f "$IN_MODEL_PATH_S"; then
    echo 'start convert model: ' $IN_MODEL_PATH_S
    python ./pca/tool_convert_torch_to_tf.py -in_model_path $IN_MODEL_PATH_S -out_model_path $OUT_MODEL_PATH_S  -vic_mesh_path $VIC_MESH_PATH
else
    echo 'something wrong. the model path does not exist: ' $IN_MODEL_PATH_S
fi


echo '\n\nconvert front pytorch pretrained weight to tensorflow graph'
IN_MODEL_PATH_F="$DATA_DIR/models/f/final_model.pt"
OUT_MODEL_PATH_F="$DATA_DIR/models/shape_model_f.jlb"

if test -f "$IN_MODEL_PATH_F"; then
    echo 'start convert model: ' $IN_MODEL_PATH_F
    python ./pca/tool_convert_torch_to_tf.py -in_model_path $IN_MODEL_PATH_F -out_model_path $OUT_MODEL_PATH_F  -vic_mesh_path $VIC_MESH_PATH
else
    echo 'something wrong. the model path does not exist: ' $IN_MODEL_PATH_F
fi
