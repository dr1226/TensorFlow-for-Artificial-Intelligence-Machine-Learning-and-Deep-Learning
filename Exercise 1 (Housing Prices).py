import tensorflow as tf
import numpy as np
from tensorflow import keras

#need to model the linear function y = 50x + 50
def house_model (y_new):
    xs = np.array([0,1,2,3,4,5,6], dtype=float)
    ys = np.array([50,100,150,200,250,300,350], dtype=float)
    model = tf.keras.Sequemtial([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new)

prediction = house_model([7.0])
print(prediction)
