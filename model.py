from random import random
import numpy as np
from PIL import Image
import PIL.ImageOps
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
import os, ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


x,y = fetch_openml('mnist_784', version=1, return_X_y=True)

x_train, x_test, y_train, y_test = tts(x, y, random_state=4, train_size=7500, test_size=2500)
x_train = x_train/255.0
x_test = x_test/255.0

model = LogisticRegression(solver = "saga", multi_class= "multinomial")
model.fit(x_train, y_train)


def prediction(image):
    img_pil = Image.open(image)
    img_bw = img_pil.convert('L')
    img_resized = img_bw.resize((28, 28), Image.ANTIALIAS)
    min_pix = np.percentile(img_resized, 20)
    img_clipped = np.clip(img_resized-min_pix, 0,255)
    max_pix = np.max(img_resized)
    img_clipped = np.asarray(img_clipped)/max_pix

    test_sample = np.array(img_clipped).reshape(1, 784)
    test_pred = model.predict(test_sample)
    return test_pred[0]

