import tensorflow as tf
import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
seed=25
tf.random.set_seed(seed)

import time

def run_adversarial_attack(model, valid_dataset , attack, attack_params=None, **kwargs):
    """Runs only the adversarial attack on a trained model. 
    
    Differs from standard training by skipping the gradient updates which can be highly costly. Can be
    demonstrated from training model in standard way and then applying"""
    # Convert function to tf function
    attack_fn = tf.function(attack)

    t = time.time()
    test_losses = []
    test_accs = []
    test_ious = []
#     print(valid_dataset.length())
    for Xtest, ytest in valid_dataset:         
        # Run attack perturbation
        if attack_params is not None:
            delta = attack_fn(model, Xtest, ytest, **attack_params)
        else:
            delta = attack_fn(model, Xtest, ytest, **kwargs)
        
#         print(delta)
        Xdtest = Xtest + delta
        l, IOU, acc = model.test_on_batch(Xdtest, ytest)
        # print(l)
        test_losses.append(l)
        test_accs.append(acc)
        test_ious.append(IOU)
    
    test_loss = sum(test_losses) / len(test_losses)
    test_acc = sum(test_accs) / len(test_accs)
    test_iou = sum(test_ious) / len(test_ious)
    logging.info(f"Time: {(time.time()-t):0.2f} Test Loss: {test_loss:0.2f}, Test Acc: {test_acc:0.2f}, Test IOU :{test_iou:0.3f}")
