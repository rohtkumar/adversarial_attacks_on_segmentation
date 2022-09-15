import argparse
import importlib
import os
import sys
import logging
import ast
import tensorflow as tf
import inspect
from util import tools


def parse_arguments():

    parser = argparse.ArgumentParser()
    add = parser.add_argument
    
    add("--batch_size", type=int, default=1)
    add("--checkpoint", default=None)
    add("--cuda", default=True)
    add("--evaluation", default=False)
    add("--num_workers", type=int, default=8)
    add("--save", default="temp_exp/", type=str)
    add("--start_epoch", type=int, default=1)
    add("--total_epochs", type=int, default=1)    
    add("--lr_scheduler", type=tools.str2str_or_none)
    #add("--lr_scheduler_values", type=tools.str2intlist)
    #add("--lr_scheduler_boundries", type=tools.str2intlist)
    add("--img_size", type=int, default=128)
    add("--optimizer_lr", type=float)
    add("--model_name", type=tools.str2str_or_none)
    add("--early_stopping", type=tools.str2bool, default=True)
    add("--early_stopping_patience", type=int, default=2)
    add("--dataset_root", default="/", type=str)
    add("--meta_args", type=int, default=6)
    add("--source_home", type=tools.str2str_or_none)
    add("--file_name", default="meta.csv", type=tools.str2str_or_none)
    add("--finetuning", type=tools.str2bool, default=False)
    add("--train", type=tools.str2str_or_none, default="train")
    add("--training_augmentation", type=tools.str2bool, default=True)
    add("--validation_augmentation", type=tools.str2bool, default=False)
    add("--model", type=tools.str2str_or_none)    
    add("--optimizer", type=tools.str2str_or_none, default='SGD')
    add("--training_loss1", type=tools.str2str_or_none, default='DiceLoss')
    add("--training_loss2", type=tools.str2str_or_none, default='CategoricalFocalLoss')
    add("--validation_loss", type=tools.str2str_or_none, default='CategoricalFocalLoss')   
    
    args = parser.parse_args()

    return args

def add_arguments_for_module(parser, module_name, default_class ):
    
    module = importlib.import_module(module_name)
    class_ = getattr(module, default_class)
    print(class_) 
