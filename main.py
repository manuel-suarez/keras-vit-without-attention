# Setup and imports
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa

# Setting seed for reproducibiltiy
SEED = 42
keras.utils.set_random_seed(SEED)

print(tf.__version__)