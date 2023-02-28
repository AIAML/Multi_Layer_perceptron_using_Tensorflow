 <h2> Classification With Multi Layer Perceptron Using Tensorflow </h2>
 
 <p> In order to classify your doncuments using Tensowrflow by Python you need to import the following Packages. </p>
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
 

 
 
 
