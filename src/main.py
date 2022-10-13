import os
import logging
import colorama
from util import commandline as cl, logger, tools
from data import make_dataset as md
from models import models, Std_train, adversarial_train as adv_train
from attacks import attacks as at
from features import build_robustifiers as robust
import segmentation_models as sm
import tensorflow


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    
    tools.set_seed(25)
    train_dataset, val_dataset = md.get_dataset(args.dataset_root, args.img_size, args.batch_size, 500)
    # train_dataset, val_dataset, test_dataset = md.get_cityscape_dataset(args.dataset_root, args.img_size,
    #                                                                     args.batch_size, 500)

    with logger.LoggingBlock("Setup Hyperparameters", emph=True):
        class_ = tools.getClass("segmentation_models.losses", args.training_loss1)
        args.training_loss1 = class_()
        class_ = tools.getClass("segmentation_models.losses", args.training_loss2)
        args.training_loss2 = class_()
        class_ = tools.getClass("tensorflow.keras.optimizers", args.optimizer)
        args.optimizer1 = class_
        class_ = tools.getClass("tensorflow.keras.optimizers.schedules", args.lr_scheduler)
        args.lr_scheduler1 = class_

    with logger.LoggingBlock("Setup Model", emph=True):
        if args.mode == "std":
            args.std_model = models.initialize_std_model(args, 10, "softmax")
            Std_train.train(args, train_dataset, val_dataset)
        elif args.mode == "adv":
            args.adv_model = models.init_adv_model(args, 10, "softmax")
            adv_train.adversarial_training(args, train_dataset, val_dataset, train_attack=at.pgd_l2_adv, epsilon=0.5, num_iter=7, alpha=0.5 / 5, epochs=63)
        elif args.mode == "robustifier":
            args.adv_model = tools.load_model(models.init_adv_model(args, 10, "softmax"), args.load)
            logging.info("Loading saved model")
            robust_model = models.get_robust_model(args)
            robust.robustify(args, robust_model, train_dataset, iters=1000, alpha=0.1)
        elif args.mode == "std_test":
            args.std_model = models.initialize_std_model_test(args, 10, "softmax")
            robust_tr = tools.get_dataset(args.load)
            logging.info(f'Loaded robust training dataset of lenght {robust_tr}')
            Std_train.train(args, robust_tr, val_dataset)
        else:
            print(args.model)



if __name__ == "__main__":
    main()

#../data/resnet50adv20221012-133716/adv_resnet50_unet.h5