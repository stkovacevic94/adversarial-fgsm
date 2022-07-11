from tensorflow.examples.tutorials.mnist import input_data
from neural_network import NeuralNetwork
import numpy as np

mnist_data = input_data.read_data_sets("MNIST_data", one_hot=True)

k = NeuralNetwork(c=[400, 100])
k.initialize(28*28, 10)

# Perform stohastic gradient descent
for _ in range(30000):
    batch = mnist_data.train.next_batch(50)
    k.run_epoch(batch[0], batch[1], learning_rate=0.001)

# Accuracy
path = "MODEL_DATA/mnist_custom/model.pkl"

k.save(path)
k = NeuralNetwork.load(path)

Y_prim = k.forward(mnist_data.test.images)
preds = np.argmax(Y_prim, axis=1)
trues = np.argmax(mnist_data.test.labels, axis=1)
accuracy = np.sum(preds == trues)/(float(len(trues)))
print("Test set accuracy:", accuracy)
