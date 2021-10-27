EXP_ID='1'
DATASET_NAME='coco'
DATA_PATH='/data1/weihao2/data/'${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} \
  --logger_name runs/${DATASET_NAME}_butd_region_bert_${EXP_ID}/log --model_name runs/${DATASET_NAME}_butd_region_bert_${EXP_ID} \
  --num_epochs=25 --lr_update=15 --learning_rate=.0005 --precomp_enc_type basic --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1
