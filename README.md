# kaggle-store-sales

*May 5, 2023*

## Contributors:
- Eoin Flanagan
- Griffin Reichert
- Leo-Paul Caucheteux
- Yudhis Lumadyo


Repository for the [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) Kaggle Competition


## Instructions for use

### Conda

There is an `environment.yml` file in the root of the directory used to set up a conda environment. To create the environement, run:
```bash
conda env create -f environment.yml
```

The environment can be activated with the command:
```bash
conda activate ibex
```

If you have added required packages to the conda environment, it can be updated using
```bash
conda env export > environment.yml
```

### Kaggle

To use the kaggle command line API tool, you must set up your kaggle account with an api token. The [docs](https://www.kaggle.com/docs/api) are helpful for this. This file should never go on git, it is specific to you and your machine.

Basic Steps TLDR:
1. Go to `My Account` on kaggle
2. Click `Create New API Token` to download a new token to your machine
3. Move the `kaggle.json` file from your downloads to:
    - OSX/Linux: `~/.kaggle/kaggle.json`
    - Windows: `C:\Users<Windows-username>.kaggle\kaggle.json`

If you did that correctly, you should be able to run
```bash
kaggle competitions list
```

### Folder Structure

[Cookiecutter Suggestions](https://drivendata.github.io/cookiecutter-data-science/)
```markdown
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```