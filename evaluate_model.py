import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, seaborn as sn; sn.set(font_scale=1.4)
import os, cv2
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = tf.keras.models.load_model('catdog_model_44_epchs(cats_retrain).h5') # saved model

class_names = ['cats','dogs']
class_name_labels = {class_name : i for i, class_name in enumerate(class_names)}


print("*"*140,"\n\n")

img = tf.keras.utils.load_img('dataset1/Single_prediction/cat7.jpg', target_size=(180,180))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"The image is {class_names[np.argmax(score)]} with {100*np.max(score)} percent confidence")