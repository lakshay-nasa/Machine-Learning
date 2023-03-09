import sys
assert sys.version_info >= (3,5)

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os



# To plot the figures

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "house_rent_forecast"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)



# Analyzing the Data

import pandas as pd
import tarfile
import urllib.request

def load_housing_data(housing_path="./House_Rent_Dataset.csv"):
    return pd.read_csv(housing_path)


housing_rent = load_housing_data()
print(housing_rent.head())

# print(pd.read_csv("./House_Rent_Dataset.csv"))
