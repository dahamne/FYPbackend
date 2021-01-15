# This program is wriite to randomly get a data from

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras

# teset data set and predict the value


print("imports successful")

#get the testing data set##
#test_setx = np.load("test_setx.npy")
#test_setx = test_setx.reshape(336, 80, 104, 1)
#print("done Loading Data")


# Creating a Sequential model
regressor = keras.models.Sequential()
regressor.add(keras.layers.Conv2D(kernel_size=(3, 3), filters=64,
                                  activation='relu', input_shape=(80, 104, 1)))
regressor.add(keras.layers.Conv2D(
    filters=50, kernel_size=(3, 3), activation='relu'))
regressor.add(keras.layers.Dropout(0.3))
regressor.add(keras.layers.MaxPool2D(2, 2))
regressor.add(keras.layers.Conv2D(
    filters=30, kernel_size=(3, 3), activation='relu'))
regressor.add(keras.layers.Flatten())

regressor.add(keras.layers.Dense(100, activation='relu'))
regressor.add(keras.layers.Dense(12, activation='softmax'))
print("model is created")

loss_fn = keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0,
    reduction="auto",
    name="categorical_crossentropy",
)

# define optimizers & compile the model
optimizer = keras.optimizers.Adam()
regressor.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

print("Successfully created the mdoel..............")

##############Loadin the Data#############################################
# import data drom the drive
path = 'Data'
training_setx = np.load(path+"/training_setx.npy")
training_sety = np.load(path+"/training_sety.npy")
training_sety = keras.utils.to_categorical(training_sety)


test_setx = np.load(path+"/test_setx.npy")
test_sety = np.load(path+"/test_sety.npy")
test_sety = keras.utils.to_categorical(test_sety)

print("Import Data successfull")

training_setx = training_setx.reshape(2400, 80, 104, 1)  # Reshaping the data
test_setx = test_setx.reshape(336, 80, 104, 1)

print("Training the mode.....")

history = regressor.fit(training_setx, training_sety, epochs=1, batch_size=32,
                        validation_data=(test_setx, test_sety))

print("Training the model Done => Saving the model")

regressor.save("finalmodel.h5")
