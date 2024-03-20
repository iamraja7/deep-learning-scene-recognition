import numpy as np
import tensorflow as tf
import os

from keras.utils.np_utils import to_categorical
from layers.Activation import Activation, SoftmaxActivation, ReluActivation
from layers.Conv2D import Conv2D
from layers.ConvNeuralNetwork import NeuralNetwork
from layers.Dense import Dense
from layers.Flatten import Flatten
from layers.Pooling import MaxPooling2D
from layers.utils import AdamOptimizer, CalCrossEntropy, SquaredLoss


def download_dataset():
    os.system("pip install kaggle terminaltables")
    os.system("mkdir ~/.kaggle")
    os.system("cp kaggle.json ~/.kaggle/")
    os.system("kaggle datasets download -d puneet6060/intel-image-classification")

    # Unzipping downloaded data
    os.system("rm -rf seg_test seg_train seg_val")
    os.system("unzip -o intel-image-classification.zip &> /dev/null")
    # Rearranging dataset folders
    os.system("mkdir seg_val && mv seg_test/seg_test/** seg_val/ && rm -rf seg_test**")
    os.system("mkdir seg_test && mv seg_pred/seg_pred/** seg_test/ && rm -rf seg_pred**")
    os.system("mv seg_train/seg_train/** seg_train/ && rm -rf seg_train/seg_train")


class DeepLearningModel:

    def __init__(self, n_inputs, n_outputs, val_datas):
        model = NeuralNetwork(opt_type=AdamOptimizer(), loss=CalCrossEntropy, val_datas=val_datas)
        model.add(Conv2D(inp_size=n_inputs, number_of_filters=16, f_size=(2, 2), stride=1, pad_val='same'))
        model.add(Activation(ReluActivation))
        model.add(MaxPooling2D(pool_shape_size=(2, 2), stride=2, padding='same'))

        model.add(Conv2D(number_of_filters=32, f_size=(2, 2), stride=1, pad_val='same'))
        model.add(Activation(ReluActivation))
        model.add(MaxPooling2D(pool_shape_size=(2, 2)))  # Valid padding

        model.add(Conv2D(number_of_filters=64, f_size=(2, 2), stride=1, pad_val='same'))
        model.add(Activation(ReluActivation))
        model.add(MaxPooling2D(pool_shape_size=(2, 2)))  # Valid padding

        model.add(Conv2D(number_of_filters=128, f_size=(2, 2), stride=1, pad_val='same'))
        model.add(Activation(ReluActivation))
        model.add(MaxPooling2D(pool_shape_size=(2, 2)))  # Valid padding

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation(ReluActivation))

        model.add(Dense(256))
        model.add(Activation(ReluActivation))

        model.add(Dense(n_outputs))
        model.add(Activation(SoftmaxActivation))

        self.model = model

    def get_model(self):
        return self.model


print("------------------------------ DOWNLOADING DATASET ------------------------------")
download_dataset()

# Defining train and val directory
train_dir = './seg_train'
val_dir = './seg_val'

# Reading dataset from images
BATCH_SIZE = 32
IMG_SIZE = (154, 154)
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(class_names)

# Data Preprocessing
print("------------------------------ DATA PREPROCESSING ------------------------------")
rescale = tf.keras.layers.Rescaling(1. / 255)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds = val_ds.map(lambda x, y: (rescale(x), y))

image_batch, labels_batch = next(iter(train_ds))
first_image = image_batch[0]
print("Min and max values after rescaling:", np.min(first_image), np.max(first_image))


def get_processed_input(dataset):
    X = []
    y = []
    for image_batch, label_batch in dataset:
        for i in range(BATCH_SIZE):
            if i < image_batch.shape[0]:
                X.append(image_batch[i].numpy())
                y.append(label_batch[i].numpy())

    X = np.array(X)
    y = np.array(y)

    X = np.moveaxis(X, -1, 1)
    y = to_categorical(y.astype("int"))

    return X, y


X_train, y_train = get_processed_input(train_ds)
X_val, y_val = get_processed_input(val_ds)
print("Shape of X_train, y_train:", X_train.shape, y_train.shape)

n_epochs = 10
IMG_SHAPE = (3,) + IMG_SIZE
n_outputs = 6
model = DeepLearningModel(n_inputs=IMG_SHAPE, n_outputs=n_outputs, val_datas=(X_val, y_val)).get_model()

print("------------------------------ MODEL SUMMARY ------------------------------")
model.summary()

# Model training
print("------------------------------ MODEL TRAINING ------------------------------")
train_err, val_err, train_acc, val_acc = model.fit(X_train, y_train, nepochs=n_epochs, batch_size=BATCH_SIZE)

print("------------------------------ MODEL PERFORMANCE ------------------------------")
print("Training accuracy: {:.4f}".format(100 * train_acc[-1]))
print("Validation accuracy: {:.4f}".format(100 * val_acc[-1]))
print("Training loss: {:.4f}".format(train_err[-1]))
print("Validation loss: {:.4f}".format(val_err[-1]))
