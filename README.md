 <h2> Classification With Multi Layer Perceptron Using Tensorflow </h2>
 
 <p> In order to classify your doncuments using Tensowrflow by Python you need to import the following Packages. </p>
 <code>
  import tensorflow as tf
 from tensorflow import keras
 import pandas as pd
 import matplotlib.pyplot as plt
 </code>
 
 <p> After that we must include our dataset. In this project we have used Fashion MNIST dataset where you can see details in <a herf='https://keras.io/api/datasets/fashion_mnist/'> this link </a> </p>
 
 fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
 
 
 
