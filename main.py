import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, seaborn as sn; sn.set(font_scale=1.4)
import os, cv2
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class_names = ['cats','dogs']
class_name_labels = {class_name : i for i, class_name in enumerate(class_names)}

number_of_classes = len(class_names)
print(class_name_labels)
IMAGE_SIZE = (180, 180)

def load_data():
    DIRECTORY = r"D:\AI\FlowerRecognition\FlowerRecognition\dataset1"
    CATEGORY = ['training_set', 'test_set']
    
    output = []
    
    for category in CATEGORY:
        path = os.path.join(DIRECTORY, category)
        images = []
        labels = []
        
        print('loading {}'.format(category))
        
        for folder in os.listdir(path):
            label = class_name_labels[folder]
            
            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(os.path.join(path, folder), file)
                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)
                
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')
            
        output.append((images, labels))
        
    return output

(training_images, training_labels), (test_images, test_labels) = load_data()
training_images, training_labels = shuffle(training_images, training_labels, random_state = 25)

# model = tf.keras.models.load_model('catdog_model_11_epchs.h5') # saved model

# uncomment to train the model

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(180,
                                  180,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

num_classes = len(class_names)

model = Sequential([
     
  data_augmentation,

  layers.Rescaling(1./255),
  
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  

  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary
epochs = 44
history = model.fit(training_images, training_labels, batch_size = 128, epochs = epochs, validation_split= 0.2)

acc = history.history['accuracy']
val_acc =  history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('catdog_model_44_epchs(cats_retrain).h5') #  epch

result = model.evaluate(test_images, test_labels, batch_size=128)

print("\nModel.evaluate ================> \n")
print(result)

print("*"*140,"\n\n")

img = tf.keras.utils.load_img('dataset1/Single_prediction/cat5.jpg', target_size=(180,180))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"The image is {class_names[np.argmax(score)]} with {100*np.max(score)} percent confidence")