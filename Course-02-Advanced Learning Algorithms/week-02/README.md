# Neural network training
## Neural Network Training
### TensorFlow implementation
![Neural Network](./images/neural-network-01.PNG)
Given set of $(x,y)$ examples, how to build and train neural network in the code?
1. Ask TensorFlow to string together the neural network layers:
    ```python
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    model = Sequential(
        [               
            tf.keras.Input(shape=(400,)),
            Dense(25, activation='sigmoid', name = 'layer1'),
            Dense(15, activation='sigmoid', name = 'layer2'),
            Dense(1, activation='sigmoid', name = 'layer3')
        ], name = "my_model" 
    )
    ```
2. Ask TensorFlow to compile together the model and specify the _Loss_ function (in this case we use `BinaryCrossentropy` loss function)
    ```python
    from tensorflow.keras.losses import BinaryCrossentropy

    model.compile(loss=BinaryCrossentropy())
    ```
3. Call the `fit` function to fit the model to the data set `X, Y` and specify how many steps in the gradient descent `epochs`
    ```python
    model.fit(X, y, epochs=100)
    ```

[<<Previous](../week-01/README.md) | [Next>>]()