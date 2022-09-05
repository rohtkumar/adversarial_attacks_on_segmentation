import os
import logging
import colorama
import pandas as pd
from core import dataloader as loader, cleanDataframe as cldf
from utils import commandline as cl, logger, tools, utils
from models import model
from common import trainer


def main():
   
    root_home = os.path.dirname(os.path.realpath(__file__))
    #print("file path %s" %root_home)
#     os.chdir(os.path.dirname(os.path.realpath(__file__)))
    train_loader = None
    test_loader = None
    args = cl._parse_arguments()
    args.source_home = root_home
    args.save = root_home+args.save
    args.num_workers = args.num_workers if args.cuda else 0
    if not os.path.exists(args.save):
        x = os.makedirs(args.save, mode=0o777, exist_ok=True)
    os.chdir(args.save)    
    logger.configure_logging(os.path.join(args.save, 'logbook.txt'))
    
    tools.write2file(sorted(vars(args).items()), filename=os.path.join(args.save, 'args.txt'))   
    
    tools.set_seed(42)
    train_dataloader, val_dataloader, test_dataloader = loader.configure_data_loaders(args)
    
    
    
    
    

    for i in range(args.kfold_num):
        with logger.LoggingBlock("Setup Hyperparameters", emph=True):  
            class_ =  tools.getClass("sm.losses", args.training_loss1)
            args.training_loss1 = class_()
            class_ =  tools.getClass("sm.losses", args.training_loss2)
            args.training_loss2 = class_()
            #class_ =  tools.getClass("torch.nn", args.validation_loss)
            #args.validation_loss1 = class_()
            class_ =  tools.getClass("torch.optim", args.optimizer)
            args.optimizer1 = class_
            class_ =  tools.getClass("torch.optim.lr_scheduler", args.lr_scheduler)
            args.lr_scheduler1 = class_
        
        with logger.LoggingBlock("Setup Model", emph=True): 
            args.stdmodel = model.initialize_std_model(args.model, 10, "softmax")
            args.optimizer1 = args.optimizer1(args.model.parameters(), args.optimizer_lr)
            args.lr_scheduler1 = args.lr_scheduler1(args.optimizer1, args.lr_scheduler_milestones, args.lr_scheduler_gamma)
        if args.train is True:
            trainer.train_model(args, train_dataloader[i], val_dataloader[i], test_dataloader[i], i+1)    
           
        else:
#             print(args.model)
            if i==4:
                trainer.evaluate(args, test_dataloader[i], i+1)
    utils.get_results_plots(args)

if __name__ == "__main__":
    main()
