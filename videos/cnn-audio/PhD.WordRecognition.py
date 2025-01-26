import librosa
import os

from preprocess import *
import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras._tf_keras.keras.utils import to_categorical
import wandb
from wandb.integration.keras.callbacks.tables_builder import WandbEvalCallback as WandbCallback
from wandb.integration.keras.callbacks.metrics_logger import WandbMetricsLogger
import matplotlib.pyplot as plt
import sys

from org_preprocess import *

from keras._tf_keras.keras.utils import to_categorical
from wandb.integration.keras.callbacks.metrics_logger import WandbMetricsLogger
import tensorflow as tf
import glob
import wandb
print("System Version::",sys.version)
print("")
print("**********************************************************************************")
print("Chandra Sen Bhagan. PhD Artificial Intelligence & Sanskrit. Start of program")
print("**********************************************************************************")


# Set hyper-parameters
wandb.init()
config = wandb.config
config.max_len = 11
config.buckets = 13


# Cache pre-processed data
if len(glob.glob("*.npy")) == 0:
    save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels = [
    "ai_u_Ṇ", "ṛ_ḷ_K", "e_o_Ṅ", "ai_au_C", "ha_ya_va_ra_Ṭ", "la_Ṇ", 
    "ña_ma_ṅa_ṇa_na_M", "jha_bha_Ñ", "gha_ḍha_dha_Ṣ", "ja_ba_ga_ḍa_da_Ś", 
    "kha_pha_cha_ṭha_tha_ca_ṭa_ta_V", "ka_pa_Y", "śa_ṣa_sa_R", "ha_L"
]
num_classes = len(labels)  # 14

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
config.epochs = 400
config.batch_size = 100


X_train = X_train.reshape(
    X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(
    X_test.shape[0], config.buckets, config.max_len, channels)

y_train_hot = tf.keras.utils.to_categorical(y_train)
y_test_hot = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(
    config.buckets, config.max_len, channels)))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
config.total_params = model.count_params()


model.fit(X_train, y_train_hot,epochs=config.epochs, batch_size=config.batch_size, validation_data=(X_test, y_test_hot), callbacks=[WandbMetricsLogger()]) 


