#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:./"
DATA_DIR=$1
TARGET_DIR="${DATA_DIR}/target/"
HEIGHT_PATH="${DATA_DIR}/height.txt"
PCA_MODEL_PATH="${DATA_DIR}/pca_model.jlb"

echo 'DATA_DIR = ' $DATA_DIR
echo 'TARGET_DIR = ' $TARGET_DIR
echo 'HEIGHT_PATH = ' $HEIGHT_PATH
echo 'PCA_MODEL_PATH = ' $PCA_MODEL_PATH

echo 'start training front model'

python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type f -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH    \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 30

echo 'start training side model'
python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type s -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH    \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 30

echo 'start training joint model'
python ./pca/nn_vic_train.py -root_dir $DATA_DIR -target_dir $TARGET_DIR -model_type joint -height_path $HEIGHT_PATH -pca_model_path $PCA_MODEL_PATH    \
        -is_scale_target 1  \
        -is_scale_height 1 \
        -use_height 1 \
        -use_gender 1 \
        -num_classes 51 \
        -n_epoch 30

echo 'convert pytorch pretrained weight to tensorflow graph'
IN_MODEL_PATH="$DATA_DIR/models/joint/final_model.pt"
OUT_MODEL_PATH="$DATA_DIR/models/joint/shape_model.jlb"

if test -f "$IN_MODEL_PATH"; then
    echo 'start convert model: ' $IN_MODEL_PATH
    python ./pca/tool_convert_torch_to_tf.py -in_model_path $IN_MODEL_PATH -out_model_path $OUT_MODEL_PATH
else
    echo 'something wrong. the model path does not exist: ' $IN_MODEL_PATH
fi