import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('./model.h5')

img_path = './dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)


from tensorflow.keras.applications.mobilenet import preprocess_input

img_processed = preprocess_input(img_batch)
prediction = model.predict(img_processed)
print(prediction)
print(type(prediction))