import os
import logging
import colorama
import pandas as pd
from util import commandline as cl, logger, tools
from data import make_dataset as md
from models import models, Std_train as train
import segmentation_models as sm


def main():
   
    root_home = os.path.dirname(os.path.realpath(__file__))
    # print("file path %s" %root_home)
#     os.chdir(os.path.dirname(os.path.realpath(__file__)))
    train_loader = None
    test_loader = None

    args = cl.parse_arguments()
    print(args.save)
    args.source_home = root_home
    # args.save = root_home+args.save

    if not os.path.exists(args.save):
        os.makedirs(args.save, mode=0o777, exist_ok=True)
    os.chdir(args.source_home)
    logger.configure_logging(os.path.join(args.save, 'logbook.txt'))

    tools.write2file(sorted(vars(args).items()), filename=os.path.join(args.save, 'args.txt'))
    with logger.LoggingBlock("Intialization", emph=True):
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Fore.GREEN
            color = colorama.Fore.CYAN
            logging.info('{}{}: {}{}'.format(color, argument, value, reset))
    
    tools.set_seed(42)

    train_dataset, val_dataset = md.get_dataset(args.dataset_root, 128, args.batch_size, 12)

    with logger.LoggingBlock("Setup Hyperparameters", emph=True):
        class_ = tools.getClass("sm.losses", args.training_loss1)
        args.training_loss1 = class_()
        class_ = tools.getClass("sm.losses", args.training_loss2)
        args.training_loss2 = class_()
        class_ = tools.getClass("tensorflow.keras.optimizers", args.optimizer)
        args.optimizer1 = class_
        class_ = tools.getClass("tf.keras.optimizers.schedules", args.lr_scheduler)
        args.lr_scheduler1 = class_

    with logger.LoggingBlock("Setup Model", emph=True):
        args.stdmodel = models.initialize_std_model(args.model, 10, "softmax")
        args.optimizer1 = args.optimizer1(args.model.parameters(), args.optimizer_lr)
        args.lr_scheduler1 = args.lr_scheduler1(args.optimizer1, args.lr_scheduler_milestones, args.lr_scheduler_gamma)

        if args.train is True:
            train.train(args, train_dataset, val_dataset)
           
        else:
#             print(args.model)
            if i==4:
                trainer.evaluate(args, test_dataloader[i], i+1)
    utils.get_results_plots(args)

if __name__ == "__main__":
    main()
