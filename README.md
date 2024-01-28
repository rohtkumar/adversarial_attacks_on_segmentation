
# Adversial Attacks on Segmentation Models
==============================

The work here describes the adversarial attacks on the deep learning models used in segmentation tasks for off road environment and also provides the implmenatation in Python3, tensorflow and keras. The model highlighted here provides the segmentated images after different attacks on the dataset and its robustness against these adversarial attacks.

The repository further includes:
	1. Source code of the model trained against L2 and L inf attacks.
	2. Modified dataset after each training.
	3. Pretrained weights for the models after each training epoch for both L2 and Linf attacks.   

#Traning on your own forest dataset

This project is trained on forset dataset consisting of opensource yamaha forest and frieburg dataset on the pretrained model weights.

To train on your own dataset follow as follows:
	1. Make a folder named "dataset" inside the main repository 
	2. Create folder as images and mask to store images and its labeled value in jpg format.
	3. With the help of VIA (VGG Image Annotator), annotate your images and make sure to have a correct format. (http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html) 

# Training the Model

There are 3 different modes of traning
	1. Standard Training - To train the model without any attacks with pretrained Kitti weights.
	2. Adversial training - Traning the model with adversarial attacks.
	3. Robustifier training - Traning the model for robustifiaction on the adversarial trained model

This work is done on pretrained models trained on Kitti dataset for statndard training to get the reference.  however you have options to train on different weights

To train a model  
	1. Edit `train.sh` file and check for other details as per your requirements.
	2. Provide the model name such as efficientnetv3 or any other as first parameter
	3. Provide the training mode as second parameter.
	4. Provide the pretrained model weights, if neccessary
	4. Run the script as follows 
`./train.sh ../dataset std_train ../data/saved_trained_model_weights`

#Requirements

Python 3.7, Tensorflow 1.7, Keras 2.0.8 and other common packages listed in `requirements.txt`

#Installation

1. Clone the respository 
2. Install dependencies
`  pip3 install -r requirements.txt`
3. Run setup from repository root direcotry
`  python3 setup.py install`

# Detailed Project structure as belows: 
Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Saved model and the dataset after model run
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── attacks        <- Different attacks methods and also the robustifier
    │   │   └── attacks.py
    │   │   └── robustifier.py
    │   ├── data           <- Generate data points in required format
    │   │   └── make_dataset.py
    │   ├── features       <- Features generation for robustifier
    │   │   └── build_robustifier.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models both standard and adversarial
    │   │   ├── adversarial_train.py
    │   │   └── evaluate_attack.py
    │   │   └── models.py
    │   │   └── Std_train.py
    │   │
    │   ├── util           <- Utiliy fucntions
    │   │   ├── commandline.py
    │   │   └── logger.py
    │   │   └── parsing.py
    │   │   └── tools.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
