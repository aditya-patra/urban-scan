import os

import scipy
import tensorflow as tf
import matplotlib.pyplot as plt
import webbrowser
import cv2
import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.optimizers import RMSprop
'''
img = tf.keras.preprocessing.image.load_img("./Coin_Img/Training/Pennies/1.jpg")
print(plt.imshow(img))
image1 = cv2.imread("./Coin_Img/Training/Pennies/1.jpg")
print(image1)
'''
train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
validation = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
train_dataset = train.flow_from_directory("./Coin_Img/Training", target_size=(150, 150), batch_size=3,
                                          class_mode='categorical')
validation_dataset = validation.flow_from_directory("./Coin_Img/Validation/", target_size=(150, 150), batch_size=2,
                                                    class_mode='categorical')
print(train_dataset.class_indices)
print(validation_dataset.class_indices)
print(train_dataset.classes)
model = tf.keras.models.Sequential([
                                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
                                tf.keras.layers.MaxPool2D(2, 2),
                                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                                tf.keras.layers.MaxPool2D(2, 2),
                                tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
                                tf.keras.layers.MaxPool2D(2, 2),
                                tf.keras.layers.Flatten(),
                                #tf.keras.layers.Dense(256, activation='relu'),
                                tf.keras.layers.Dropout(0.5),  # Adding dropout for regularization
                                tf.keras.layers.Dense(10, activation='softmax')
                                    ])
checkpoint_path = "checkpoint"

# Create a callback that saves the model's weights
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch=20,
                      epochs=50,
                      validation_data=validation_dataset)
model.save("checkpoint_path")
dir_path = './Coin_Img/Testing'
for i in os.listdir(dir_path):
    print(i)
    img = tf.keras.preprocessing.image.load_img(dir_path+"//"+i, target_size=(150,150))
    X = tf.keras.preprocessing.image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    print(val)
    '''
    if str(val) == "[[1. 0. 0. 0. 0.]]":
        print('is a dime')
    elif str(val) == "[[0. 1. 0. 0. 0.]]":
        print('is a nickel')
    elif str(val) == "[[0. 0. 1. 0. 0.]]":
        print('is a penny')
    elif str(val) == "[[0. 0. 0. 1. 0.]]":
        print('is a quarter')
    else:
        print(val)
    '''
print(validation_dataset.class_indices)