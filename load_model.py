# USAGE
# python load_model.py --images malaria/testing --model saved_model.model

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
from matplotlib import pyplot as plt
import numpy as np
import argparse
import random
import cv2

MODEL_NAME = 'streetart_model.model'
MONTAGE_FILENAME = 'streetart_montage.png'
IMAGES_PATH = 'streetart/testing'

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(MODEL_NAME)

# grab all image paths in the input directory and randomly sample them
imagePaths = list(paths.list_images(IMAGES_PATH))
random.shuffle(imagePaths)
imagePaths = imagePaths[:25]

# initialize our list of results
results = []

# loop over our sampled image paths
print("[INFO] evaluating model against test set...")
for p in imagePaths:
	# load our original input image
	orig = cv2.imread(p)

	# pre-process our image by converting it from BGR to RGB channel
	# ordering (since our Keras mdoel was trained on RGB ordering),
	# resize it to 64x64 pixels, and then scale the pixel intensities
	# to the range [0, 1]
	image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (64, 64))
	image = image.astype("float") / 255.0

	# order channel dimensions (channels-first or channels-last)
	# depending on our Keras backend, then add a batch dimension to
	# the image
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# make predictions on the input image
	pred = model.predict(image)
	pred = pred.argmax(axis=1)[0]

	# an index of zero is the 'parasitized' label while an index of
	# one is the 'uninfected' label
	label = "Not street art" if pred == 0 else "Street art found"
	color = (255, 0, 0) if pred == 0 else (0, 255, 0)

	# resize our original input (so we can better visualize it) and
	# then draw the label on the image
	orig = cv2.resize(orig, (128, 128))
	cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		color, 2)

	# add the output image to our list of results
	results.append(orig)

print("[INFO] building image montage of results...")
# create a montage using 128x128 "tiles" with 4 rows and 4 columns
montage = build_montages(results, (200, 200), (5, 5))[0]
cv2.imwrite(MONTAGE_FILENAME, montage)

img = cv2.imread(MONTAGE_FILENAME)
img2 = img[:,:,::-1]
plt.imshow(img)
