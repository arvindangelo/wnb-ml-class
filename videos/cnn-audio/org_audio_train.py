from org_preprocess import *

from keras._tf_keras.keras.utils import to_categorical
from wandb.integration.keras.callbacks.metrics_logger import WandbMetricsLogger
import tensorflow as tf
import glob
import wandb

# Set hyper-parameters
wandb.init()
config = wandb.config
config.max_len = 11
config.buckets = 13


# Cache pre-processed data
if len(glob.glob("*.npy")) == 0:
    save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels = ["bed", "happy", "cat"]

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
config.epochs = 400
config.batch_size = 100

num_classes = 3

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
#model.fit(X_train, y_train_hot, validation_data=(X_test, y_test_hot), callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels)])


#model.fit(X_train, y_train_hot, batch_size=config.batch_size, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[wandb.keras.WandbCallback(data_type="image", labels=labels)])