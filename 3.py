import tensorflow as tf
from tensorflow.keras import models,layers
import numpy as np

#Loading The MNIST Data
#Data is the heart blood of machine learning.
#You must understand how it works. 
#Especialy now that you have Ai on your team and you are going
#To be a genius on it

(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()

#What we did here was create data to feed the model and data to 
#test the model on. 
#Now we must normalize it 

x_train, x_test = x_train / 255.0 , x_test / 255.0

#If you do not understand, we normlaize dats so the model that we are going to load in can understand it. As you can see here, this is going ot be the print size. 

print(f"Training data shape : {x_train.shape}")
print(f"Testing data shape : {x_test.shape}")

#As you can see here, the shapes are (60000, 28, 28) in other words, the 
#machine can handle this 

#So cool, we got our data and now we are going to move onto building the
#model

model = models.Sequential([
    layers.Reshape((28,28,1),input_shape=(28,28)),
    
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation ='softmax')
    ])

#Model Complilation
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()


#So if you are wondering what any of this actually means 
#Then you are not alone and a normal person.
#Basically, we decided to summon a model. We did this in models.Sequential. Then what we decided to do was a ton of layers do it. Why do we do this? Well let me explain. 

#So basically, each layer of this model is going to effect that original data from earlier that we transformed in it's own way. How are we going to do that? 

#Think of them as spells 
#Layers.Reshape
#Layers.Conv2D (This spell will apply very, very small filters to detect curves, etc. Imagine him as someone with a zoom ability.)
#Layers.MaxPooling(Basically, this guy is going to shrink the image to keep only the most important features of the image. What does this mean? Think ofit like a guy that kind of brushes off all the bloat of a machine or process) 
#Layers.Flatten (This guy has to do with vectors and tensors. Basically, we have to turn the data into a 1D vector. Do you remember the charts? Why? Because the machine needs it.
#Layers.Dense (This is mor ecool machine learning stuff. Basically, we are are just connecting the neurons in the network. Yeah we are kind of awesome huh?) 

#So at the end of the day, we are really just telling the computer, "Hey Detect this," "Hey pay attention to this" "Hey convert this" and "Hey set up some links here. This is how ML thinks. 

#Training The Model

model.fit(x_train, y_train, epochs=5, validation_data=(x_test,y_test))

#Evalutation
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy : {test_acc}")
