from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

# Load the model
model = load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
video = cv2.VideoCapture(0)
while True:
    ret, frame = video.read()
    cv2_im = cv2.cvtColor(frame,cv2.COLOR_BRG2RGB)
    image = Image.fromarray(cv2_im)

# Replace this with the path to your image
#image = Image.open(frame)
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
    image_array = np.asarray(image)
# Normalize the image   
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
    data[0] = normalized_image_array

# run the inference
    prediction = model.predict(data)
    print(prediction)
