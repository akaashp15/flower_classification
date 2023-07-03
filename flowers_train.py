import matplotlib.pyplot as plt
import numpy as np
import PIL
import requests
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib


#Import Data and set directory for the data
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin = dataset_url, untar = True)
data_dir = pathlib.Path(data_dir)

##Print the number of images in the dataset
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


##Can access the subset of images containing a certain name/tag
roses = list(data_dir.glob('roses/*'))
##Use PIL.Image.open to view the image
rose_0 = PIL.Image.open(str(roses[0]))

#Loader Parameters
batch_size = 32
img_height = 180
img_width = 180

##Formalize the training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

##Formalize the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = 'validation',
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

## Printing class names
class_names = train_ds.class_names
#print(class_names)

##Autotunes the value of data dynamically at runtime
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

##Keras Model
num_classes = len(class_names)

model = Sequential([
    layers.Rescaling(1./255, input_shape = (img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(num_classes)
])

##Setting framework for the loss functions/optimization of tuning
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ['accuracy'])


##This is the model framework
print(model.summary())


##Training the model for 10 epochs
epochs = 10
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = epochs
)

##Analyze results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

##Visualize training stats
# plt.figure(figsize = (8,8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label = 'Training Accuracy')
# plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
# plt.legend(loc = 'lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label= 'Training Loss')
# plt.plot(epochs_range, val_loss, label= 'Validation Loss')
# plt.legend(loc = 'upper right')
# plt.title('Training and Validation Loss')
# plt.show()

##Predict on new data
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

sunflower_img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(sunflower_img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


##Convert model to TensorflowLite Model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

##Save model to be used again
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

