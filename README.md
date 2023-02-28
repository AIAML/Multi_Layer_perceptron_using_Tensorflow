 <h2> Classification With Multi Layer Perceptron Using Tensorflow </h2>
 
 <p> In order to classify your documents using Tensowrflow by Python you need to import the following Packages. </p>
 <code>
  import tensorflow as tf
 from tensorflow import keras
 import pandas as pd
 import matplotlib.pyplot as plt
 </code>
 
 <p> After that we must include our dataset. In this project we have used Fashion MNIST dataset where you can see details in <a href='https://keras.io/api/datasets/fashion_mnist/'> this link </a> </p>
 
<code> fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
</code>
<p> Our Train dataset contains 60000 samples so as for to prevent overlearning we have set 10000 for validation set. The following code used for this purpose. </p>
<code>
  x_valid, x_train = x_train_full[50000:] / 255.0, x_train_full[:50000] / 255.0
  y_valid, y_train = y_train_full[50000:], y_train_full[:50000]
 </code>
 
 <p> Next We have to build our model. In this sample we have applied Sequential model of neural network.  </p>
 
 <code> 
model = keras.models.Sequential()
</code>
<p> The next step is building our layers. The first layer is formed based on our input. Our Input data is an image which has 28*28 dimenstion. As a consequence our code in python would be:  </p>

<code> 
  model.add(keras.layers.Flatten(input_shape=[28, 28]))
</code>
 
 <p> For selecting your model you have to consider this issue that it takes a while to be proficient in selecting your proper model. In this problem we need to classify images so we have two hidden layer which use 'relu' function. Finally for the last layer we have used softmax function with 10 point which indicates our classes.  </p>
 
 <code>
  model = keras.models.Sequential()
  model.add(keras.layers.Flatten(input_shape=[28, 28]))
  model.add(keras.layers.Dense(300, activation="relu"))
  model.add(keras.layers.Dense(100, activation="relu"))
  model.add(keras.layers.Dense(10, activation="softmax"))
 </code>
 
 <p>
 In addtion, before fitting you can view a summery of your model.
 </p>
 <code>
  model.summary()
 </code>
 <p>
 The last step is compiling and fiting our model. Pure and Simple
 </p>
 <code>
  model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
   history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))
 </code>

 
 
 
