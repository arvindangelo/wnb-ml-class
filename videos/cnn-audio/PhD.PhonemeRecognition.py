import librosa
import os

from preprocess import *
import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras._tf_keras.keras.utils import to_categorical
import wandb
from wandb.integration.keras.callbacks.tables_builder import WandbEvalCallback as WandbCallback
import matplotlib.pyplot as plt
wandb.init()
config = wandb.config

config.max_len = 11

config.buckets = 20



#Save data to array file first

save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)



labels=["bed", "happy", "cat"]
# Loading train set and test set


#TODO: Recognise phonemes.
# step one: identify and categorise phonemes
# step two: recognise phonemes