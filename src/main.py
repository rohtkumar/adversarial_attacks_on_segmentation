import os
import logging
import colorama
import pandas as pd
from util import commandline as cl, logger, tools
from data import make_dataset as md
<<<<<<< HEAD
from models import models, Std_train as train
=======
from models import models, Std_train, adversarial_train as adv_train
import attacks
from features import build_robustifiers as robust
>>>>>>> 71f5dd9197c4d1e01a17496f08cd286e7dcf6275
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
<<<<<<< HEAD
        args.stdmodel = models.initialize_std_model(args.model, 10, "softmax")
        args.optimizer1 = args.optimizer1(args.model.parameters(), args.optimizer_lr)
        args.lr_scheduler1 = args.lr_scheduler1(args.optimizer1, args.lr_scheduler_milestones, args.lr_scheduler_gamma)
=======
        if args.mode == "Std_train":
            args.std_model = models.initialize_std_model(args, 10, "softmax")
            Std_train.train(args, train_dataset, val_dataset)
        elif args.mode == "Adv_train":
            args.adv_model = models.init_adv_model(args.model, 10, "softmax")
            adv_train.adversarial_training(args, train_dataset, val_dataset, attacks.pgd_l2_adv, epsilon=0.5, num_iter=7, alpha=0.5 / 5, epochs=25000 // 391)
        elif args.mode == "Robustfier":
            args.adv_model = tools.load_model(models.init_adv_model(args.model, 10, "softmax"), args.load)
            robust_model = models.get_robust_model(args)
            robust.robustify(args, robust_model, train_ds, iters=1000, alpha=0.1)
        else:
            print(args.model)

>>>>>>> 71f5dd9197c4d1e01a17496f08cd286e7dcf6275

        if args.train is True:
            train.train(args, train_dataset, val_dataset)
           
        else:
#             print(args.model)
            if i==4:
                trainer.evaluate(args, test_dataloader[i], i+1)
    utils.get_results_plots(args)

if __name__ == "__main__":
    main()
