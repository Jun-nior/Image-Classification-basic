model=models.load_model('image_classifier.model')

# img = cv.imread('C:\Trung Main\TEST\Safeimagekit-resized-img.png')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# plt.imshow(img, cmap=plt.cm.binary)

# prediction= model.predict(np.array([img])/255)
# index=np.argmax(prediction)
# print(f'Prediction is {class_names[index]}')
# plt.show()
# # np.set_printoptions(precision=7, suppress=True)
# print(prediction) # highest probability will be the final prediction