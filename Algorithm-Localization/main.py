import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import time
import csv
import os

model = keras.models.load_model("my_model.h5")
print("loading done...")

###################################


def getLocal(x):
    with open('/home/daham/Desktop/fyp/testflask/env/PredictedData/Local_Pred.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([x])
        print(x)


##### Loadin the test data set######
test_setx = np.load("test_setx.npy")
test_setx = test_setx.reshape(336, 80, 104, 1)
print("done loading the data.....")

##############prediction############
while True:
    i = random.randint(0, 335)
    y = model.predict_classes(test_setx[i].reshape(1, 80, 104, 1))
    getLocal(str(y[0]))
    time.sleep(2)
