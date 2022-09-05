# !/bin/bash

DATA_HOME="../../dataset/images/"
SOURCE_HOME="../adversarial_attacks_on_segmentation"
BATCH_SIZE=32
if [ $1 = 'prod' ]
then
    echo "Prod"
    pip install efficientnet-pytorch
    DATA_HOME="/project/rokumar/SIIM-ISIC/"
    SOURCE_HOME="/project/rokumar/"
    FILE_NAME="meta.csv"
    BATCH_SIZE=8
else
    echo "Local"
    DATA_HOME="/rokumar/project/SIIM-ISIC/"
    SOURCE_HOME="/project/rokumar/"
    FILE_NAME="train.csv"
    BATCH_SIZE=2
fi


#resnet50 or efficientnetb3
MODEL=$2
# save best Model checkpoint path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="$SOURCE_HOME/data/$MODEL$ALIAS$TIME"
CHECKPOINT="/thesis/rokumar/Checkpoint/"

Train_Image_Augmentation=True
Train_Loss_Function="BCELoss"

Valid_Image_Augmentation=False
Valid_Loss_Function="BCELoss"


python ../src/main.py \
--batch_size=$BATCH_SIZE \
--checkpoint=$CHECKPOINT \
--lr_scheduler=MultiStepLR  \
--model=$MODEL \
--num_workers=2 \
--kfold=True \
--kfold_num=5 \
--train=True \
--model_name=$MODEL \
--lr_scheduler_gamma=0.5 \
--lr_scheduler_milestones="[23, 39, 47, 54]" \
--early_stopping=True \
--early_stopping_patience=10 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--total_epochs=10 \
--file_name=$FILE_NAME \
--dataset_root=$DATA_HOME \
--source_home=$SOURCE_HOME \
--training_augmentation=$Train_Image_Augmentation \
--training_loss=$Train_Loss_Function \
--validation_augmentation=$Valid_Image_Augmentation \
--validation_loss=$Valid_Loss_Function \
--finetuning=False \
