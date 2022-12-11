# !/bin/bash

#DATA_HOME="../../citysacpe/gtFine/"
DATA_HOME="../../dataset/images/"
SOURCE_HOME="../adversarial_attacks_on_segmentation"
BATCH_SIZE=2


#resnet50 or efficientnetb3
MODEL=$1
MODE=$2 # std_train, adv_train, robustifier, std_test, adv_test, evaluate
LOAD_PATH=$3
# save best Model checkpoint path
TIME=$(date +"%Y%m%d-%H%M%S")
SAVE_PATH="../data/$MODEL$MODE$TIME"
CHECKPOINT="/thesis/rohkumar/Checkpoint/"

Train_Image_Augmentation=True
Train_Loss_Function1="DiceLoss"
Train_Loss_Function2="CategoricalFocalLoss"

Valid_Image_Augmentation=False
Valid_Loss_Function="BCELoss"


python3 ../src/main.py \
--batch_size=$BATCH_SIZE \
--checkpoint=$CHECKPOINT \
--lr_scheduler="PiecewiseConstantDecay"  \
--model=$MODEL \
--num_workers=2 \
--mode=$MODE \
--model_name=$MODEL \
--img_size=128 \
--is_train=True \
--cuda_device="3,2,1,0" \
--early_stopping=True \
--early_stopping_patience=10 \
--optimizer=Adam \
--optimizer_lr=1e-4 \
--save=$SAVE_PATH \
--load=$LOAD_PATH \
--start_epoch=1 \
--total_epochs=10 \
--dataset_root=$DATA_HOME \
--source_home=$SOURCE_HOME \
--training_augmentation=$Train_Image_Augmentation \
--training_loss1=$Train_Loss_Function1 \
--training_loss2=$Train_Loss_Function2 \
--validation_augmentation=$Valid_Image_Augmentation \
--validation_loss=$Valid_Loss_Function \
--finetuning=False \
