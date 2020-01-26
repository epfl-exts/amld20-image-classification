<img src="static/EXTS_Logo.png" width="125px" align="right">

# AMLD20 - Image Classification

Welcome! This repository contains all resources for the **Image Classification** hands-on exercise, presented during the [EPFL Extension School Workshop - Machine Learning and Data Visualization](https://appliedmldays.org/workshops/epfl-extension-school-workshop-machine-learning-and-data-visualization) at the [Applied Machine Learning Days 2020](https://appliedmldays.org/).

In this hands-on exercise, participant are tasked to train their own image classifier. The goal of this hands-on exercise is to provide a general overview about the topic of image classifiation, and to showcase in a practical way the moving parts of this so called machine learning "black box". At the end, the participants will be able to train their own image classification model, trained on their own chosen classes, and to predict the most likely class membership of any new image.

**Slides**: The Google slides connected to this talk can be found [here](https://docs.google.com/presentation/d/1Jg9rO_3dXwKzJyDOr2ley8Is5oWKE6D_aJJlJrpw0mw/present?usp=sharing).

## Run Hands-On in the Cloud

The most straightforward way to run this hands-on exercise is to execute it directly in your browser, i.e. in the cloud.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/epfl-exts/amld20-image-classification/blob/master/AMLD20_image_classification.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/epfl-exts/amdl20-image-classification/master?filepath=AMLD20_image_classification.ipynb)
[![Generic badge](https://img.shields.io/badge/Offline_View-Open-Blue.svg)](https://nbviewer.jupyter.org/github/epfl-exts/amdl20-image-classification/blob/master/static/AMLD20_image_classification.ipynb)

Given the computational demands of this hands-on exercise, we recommend to run it directly via **Google's Colab** feature. Should you not be able to do so, you might want to try out **Binder**. If both of these things fail, you can also take a look at the already executed notebook in the **Offline View**.

## Run Hands-On locally on your machine

Should you prefer to run the hands-on locally on your machine, either install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) on your system and use the provided [environment.yml](https://github.com/epfl-exts/amdl20-image-classification/blob/master/environment.yml) file, or use your python environment of chose and use the [colab-requirements.txt](https://github.com/epfl-exts/amdl20-image-classification/blob/master/colab-requirements.txt) file to install the required Python dependencies with `pip`.

#### 1. Clone repository content from Github

First things first, download the content of the github repository either manually via the green [Clone or download](https://github.com/epfl-exts/amdl20-image-classification/) button on the top right of the homepage, or use a terminal and run the code:

```
git clone https://github.com/epfl-exts/amdl20-image-classification.git
```

Once the content of the repository is on your machine, you can install the relevant Python dependencies with `conda` or `pip`.

#### 2a. Installation with conda

To install the relevant Python dependencies with conda, use the following code. ***Note***: This assumes that the downloaded github repository was stored in your home folder.

```
conda env create -f ~/amdl20-image-classification/environment.yml
```

#### 2b. Installation with `pip`

To install the relevant Python dependencies with `pip`, use the following code. ***Note***: This assumes that the downloaded github repository was stored in your home folder.

```
pip install -r ~/amdl20-image-classification/colab-requirements.txt
```
