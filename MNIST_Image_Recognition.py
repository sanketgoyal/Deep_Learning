#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist #To get access to MNIST Dataset of keras
from keras.models import Sequential #Allowing us to define our Neural model
from keras.layers import Dense #Connecting all the layers with preceding layers to develop fully connected network
from keras.optimizers import Adam #Dealing with multi class dataset
from keras.utils.np_utils import to_categorical #Allows taking labels as inputs and reformated them into One Hot Encoded form
import random


# In[1]:


np.random.seed(0)


# In[ ]:


(X_train, y_train),(X_test, y_test) = mnist.load_data() #Loading data from mnist dataset in form of tuple, thus have(), ()


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


#To verify the data is consisitent or each image has labels
assert(X_train.shape[0] == y_train.shape[0]), "The number if images is not equal to number of labels."
assert(X_test.shape[0] == y_test.shape[0]), "The number if images is not equal to number of labels."

#To verify the image size 28*28 pixels
assert(X_train.shape[1:] == (28,28)), "The Dimensions of images are not 28*28"
assert(X_test.shape[1:] == (28,28)), "The Dimensions of images are not 28*28"


# In[ ]:


#Plot and analyze the data to visualize each class if images present between 0 to 9
num_of_samples = [] #Amount of images in each of the ten categories
cols = 5 
num_classes = 10

fig, axs = plt.subplots(nrows = num_classes, ncols = cols, figsize = (5,10))
fig.tight_layout()

#Create a nested for loop

for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j] #Take all data stored in X_train but keep the data with labels with j only
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap = plt.get_cmap("gray"))
        
        axs[j][i].axis("off") #To not display axis
        
        if i == 2:
            axs[j][i].set_title(str(j))
            
            num_of_samples.append(len(x_selected))


# In[ ]:



print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")


# In[ ]:


#Perform one hot encoding

y_train = to_categorical(y_train, 10) #Labelled data that needs to be encoded, total number of classes
y_test = to_categorical(y_test, 10)


# In[ ]:


#Normalization (Choose to divde by 255 because we want the value between 0-1 and the max pixel value is 255)
#Scales data to uniform range and decreases the variance of the data

X_train = X_train/255
X_test = X_test/255


# In[ ]:


#Flatten the array to make images 1D for feedforward process
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)


# In[ ]:


'''
def create_model():
    model = Sequential() #To define our model
    model.add(Dense(10, input_dim = num_pixels, activation="relu")) #To add hidden layers with as many nodes in each layer
                                        #One hidden layer with 10 nodes, preceding with input layer with input = pixels
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation = "relu"))
    model.add(Dense(num_classes, activation="softmax")) #Output layer, softmax converts everything to probabilities
    model.compile(Adam(lr = 0.01), loss = "categorical_crossentropy", metrics = ["accuracy"]) 
    
    return model

#Increasing the number of nodes in a layer, will train the model pretty well. The training set will be memorized rather than learn
'''


# In[ ]:


def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


model = create_model()
print(model.summary())


# In[ ]:


#To fit our training data
history = model.fit(X_train, y_train, validation_split = 0.1, epochs = 10, batch_size = 200, verbose = 1, shuffle = 1)
#verbose is to display progress bar of info
#Depenging upon epochs, validation loss and training loss are changed
#Higher epoch like more than 30 gives more validation loss, meaning training set is perfectly fir but validation error is not
#Lower epoch like 2, makes them very far away


# In[ ]:


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.title("Loss")
plt.xlabel("epoch")


# In[ ]:


#Plotting the accuracy

plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.legend(["acc", "val_acc"])
plt.title("Acc")
plt.xlabel("epoch")


# In[ ]:


#To test on new unlabelled data
score = model.evaluate(X_test, y_test, verbose = 0)
print(type(score))
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


import requests
from PIL import Image
url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream = True)
img = Image.open(response.raw)
plt.imshow(img)



# In[ ]:


import cv2
img_array = mp.asarray(img) #converts input data as array
resized = cv2.resize(img_array, (28,28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray_scale)
plt.imshow(image, cmap = plt.get_cmap("gray"))


# In[ ]:


image = image/255
image = image.reshape(1, 784)


# In[ ]:


prediction = model.predict_classes(image)
print("Predicted digit:", str(prediction))

