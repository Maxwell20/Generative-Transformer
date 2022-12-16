# Generative-Transformer
Example of a Generative-Transformer Machine Learning Neural Network
This code loads the data, preprocesses it by encoding the characters as integers and creating training examples and targets, and then defines and trains a generative transformer model using the Keras API. The model consists of an embedding layer, two LSTM layers, and a dense layer with a softmax activation function. The model is then compiled and trained on the dataset for 10 epochs.

After training, you can use the model to generate text by providing a seed string and specifying the number of characters to generate.

is code defines a generate_text() function that takes a trained model, a start string, and the number of characters to generate as input, and generates text using the model. The start string is converted to integers and used as the input to the model. The model is then used to generate predictions for each character, and the predicted integer is converted to a character and added to the generated text. The process is repeated for the specified number of characters. The generated text is then returned as a string.
