import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import keras
from tensorflow.keras import datasets,layers,models

(training_images, training_labels),(testing_images,testing_labels)=datasets.cifar10.load_data()
training_images,testing_images=training_images/255,testing_images/255

class_names=['plane','car','bird','cat','deef','dog','frog','horse','ship','truck']

# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(training_images[i],cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i][0]])   

# plt.show();

training_images=training_images[:1000]
training_labels=training_labels[:1000]
testing_images=testing_images[:100]
testing_labels=testing_labels[:100]

# model=models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10,activation='softmax'))

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
# model_checkpoint = tensorflow.keras.callbacks.ModelCheckpoint('C:\Trung Main\TEST', monitor='val_loss', save_best_only=True)

# model.fit(training_images,training_labels,epochs=100000,validation_data=(testing_images,testing_labels), callbacks=[early_stopping,model_checkpoint])

# loss,accuracy=model.evaluate(testing_images,testing_labels)
# print(f"loss: {loss}")
# print(f"Accuracy: {accuracy}")

# model.save('image_classifier.model')
model=models.load_model('image_classifier.model')

img = cv.imread('C:\Trung Main\TEST\Horse.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction= model.predict(np.array([img])/255)
index=np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
plt.show()
np.set_printoptions(precision=5, suppress=True)
print(prediction) # highest probability will be the final prediction
