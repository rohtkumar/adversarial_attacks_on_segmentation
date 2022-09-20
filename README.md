Adversial Attacks on Segmentation Models
==============================

Adversarial Attacks on Segmentaion Networks in Off road enviornment.

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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── attacks        <- Different attacks methods and also the robustifier
    │   │   └── attacks.py
    │   │   └── robustifier.py
    │   ├── data           <- Generate data points in required format
    │   │   └── make_dataset.py
    │   ├── features       <- Features generation for robustifier
    │   │   └── build_robustifier.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models both standard and adversarial
    │   │   ├── adversarial_train.py
    │   │   └── evaluate_attack.py
    │   │   └── models.py
    │   │   └── Std_train.py
    │   │
    │   ├── util           <- Utiliy fucntions
    │   │   ├── commandline.py
    │   │   └── logger.py
    │   │   └── parsing.py
    │   │   └── tools.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
