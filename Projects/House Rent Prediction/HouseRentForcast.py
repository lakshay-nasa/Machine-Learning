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

# housing_rent.info()
# print(housing_rent["Bathroom"].value_counts())
print(housing_rent.describe())


# matplotlib inline
# import matplotlib.pyplot as plt
# housing_rent.hist(bins=50, figsize=(20,15))
# save_fig("attribute_histogram_plots")
# plt.show()


# Testing Set

np.random.seed(42)
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)* test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing_rent, 0.2)
print(len(train_set))
print(len(test_set))


