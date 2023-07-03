import tensorflow as tf
import numpy as np

from flowers_train import class_names

#Loader Parameters
batch_size = 32
img_height = 180
img_width = 180

TF_MODEL_FILE_PATH = 'model.tflite'

def flower_classification(img):
    interpreter = tf.lite.Interpreter(model_path = TF_MODEL_FILE_PATH)

    #sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    #sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    classify_lite = interpreter.get_signature_runner('serving_default')
    predictions_lite = classify_lite(rescaling_1_input = img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)
    
    return_msg = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
    return return_msg

interpreter = tf.lite.Interpreter(model_path = TF_MODEL_FILE_PATH)

sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

sunflower_img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(sunflower_img)
img_array = tf.expand_dims(img_array, 0)

print(interpreter.get_signature_list())


classify_lite = interpreter.get_signature_runner('serving_default')
predictions_lite = classify_lite(rescaling_1_input = img_array)['dense_1']
score_lite = tf.nn.softmax(predictions_lite)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)