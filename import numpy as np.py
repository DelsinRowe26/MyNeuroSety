import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from tensorflow import keras
from matplotlib import pyplot as plt

def pregprocess(img, label):
    return tf.image.resize(img, [200, 200])/255, label

split = ["train[:70%]", "train[70%:]"]

trainDataset, testDataset = tfds.load(name = 'cats_vs_dogs', split = split, as_supervised=True)

trainDataset = trainDataset.map(pregprocess).batch(32)
testDataset = testDataset.map(pregprocess).batch(32)

# model = keras.Sequential([
#     keras.layers.Conv2D(16,(3, 3), activation = 'relu', input_shape = (200, 200, 3)),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(32,(3, 3), activation = 'relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64,(3, 3), activation = 'relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(512, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# trainHistory = model.fit(trainDataset, epochs = 10, validation_data = testDataset)

# plt.plot(trainHistory.history['accuracy'])
# plt.plot(trainHistory.history['val_accuracy'])
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Training', 'Validation'])
# plt.grid()
# plt.show()

# (loss, accuracy) = model.evaluate(testDataset)
# print(loss)
# print(accuracy)

# model.save("model1.h5")
# model.save("model1.h5")
# model.save("model2.h5")

model = keras.models.load_model("model2.h5")

predictions = model.predict(testDataset.take(10))

classNames = ['cat', 'dog']

i = 0

fig, ax = plt.subplots(1, 10)

for image, _ in testDataset.take(10):
    predictedLabel = int(predictions[i] >= 0.5)

    ax[i].axis('off')
    ax[i].set_title(classNames[predictedLabel])
    ax[i].imshow(image[i])
    i += 1

plt.show()