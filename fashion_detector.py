import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import cv2 
import requests
from PIL import Image
from io import BytesIO

class Fashion:

  def __init__(self):
    super().__init__()

    train_df = pd.read_csv('fashion-mnist_train.csv',sep = ',')

    self.class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_data = np.array(train_df, dtype = 'float32')
    x_train = train_data[:,1:]/255
    y_train = train_data[:,0]

    x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2)

    image_shape = (28,28,1)
    x_train = x_train.reshape(x_train.shape[0],*image_shape)
    x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)

    self.model = Sequential([
      Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape ),
      MaxPooling2D(pool_size=2) ,
      Dropout(0.2),
      Flatten(),
      Dense(10,activation='relu'),
      Dense(10,activation = 'softmax')  
    ])

    self.model.compile(loss ='sparse_categorical_crossentropy',
                optimizer=Adam(lr=0.001),
                metrics =['accuracy'])

    self.history = self.model.fit(
    x_train,
    y_train,
    batch_size=4096,
    epochs=3,
    verbose=1,
    validation_data=(x_validate,y_validate),
    )

  ##transforme l'image pour la prédiction, de même taille que le dataset
  def preprocess_image(self, url):
      
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
      
    ## Grayscale
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
    img = img / 255.0
    ##Resize
    img = cv2.resize(img, (28,28)) 
    
    img = np.expand_dims(img, axis = [0,3])
    return img

  ##prédiction
  def predict_image(self, url): 

    img = self.preprocess_image(url)
    predicted_label = self.model.predict(img)
    predicted_label = np.argmax(predicted_label, axis = 1)[0]
    
    return self.class_names[predicted_label]
