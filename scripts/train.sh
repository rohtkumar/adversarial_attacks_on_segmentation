# !/bin/bash

DATA_HOME="../../dataset/images/"
SOURCE_HOME="../adversarial_attacks_on_segmentation"
BATCH_SIZE=4


#resnet50 or efficientnetb3
MODEL=$1
TRAIN_TYPE=$2 # Std_train, Adv_train
# save best Model checkpoint path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="../data/$MODEL$ALIAS$TIME"
CHECKPOINT="/thesis/rohkumar/Checkpoint/"

Train_Image_Augmentation=True
Train_Loss_Function1="DiceLoss"
Train_Loss_Function2="CategoricalFocalLoss"

Valid_Image_Augmentation=False
Valid_Loss_Function="BCELoss"

# https://code-with-me.global.jetbrains.com/N13olVKS8dNEt3N2s2tFMA#p=PY&fp=C5532FBB40D11F4D577D2DED1BC7416644A909454D2B82D674FC095F6CAA1EAF

python ../src/main.py \
--batch_size=$BATCH_SIZE \
--checkpoint=$CHECKPOINT \
--lr_scheduler="PiecewiseConstantDecay"  \
--model=$MODEL \
--num_workers=2 \
--train=$TRAIN_TYPE \
--model_name=$MODEL \
--img_size=128 \
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
--training_loss1=$Train_Loss_Function1 \
--training_loss2=$Train_Loss_Function2 \
--validation_augmentation=$Valid_Image_Augmentation \
--validation_loss=$Valid_Loss_Function \
--finetuning=False \
