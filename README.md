
# Adversial Attacks on Segmentation Models
==============================

The work here describes the adversarial attacks on the deep learning models used in segmentation tasks for off road environment and also provides the implmenatation in Python3 and pytorch. The model highlighted here provides the segmentated images after different attacks on the dataset and its robustness against these adversarial attacks.
The repository further includes:
	1. Source code of the model trained against L2 and L inf attacks.
	2. Modified dataset after each training.
	3. Pretrained weights for the models after each training epoch for both L2 and Linf attacks.   

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
