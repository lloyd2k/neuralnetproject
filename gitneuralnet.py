import numpy as np
import idx2numpy
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

class Sigmoid:
  def __call__(self, x):
    return 1/(1+np.exp(-x))
  
  def derivative(self, x):
    return self(x) * (1 - self(x))
  
class Relu:
   def __call__(self, x):
      return np.maximum(0, x)
   
   def derivative(self, x):
      return (x > 0).astype(float)

class Tanh:
   def __call__(self, x):
      return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
   
   def derivative(self, x):
      return 1 - self(x)**2
   
class LeakyRelu:
   def __call__(self, x):
      return np.where(x >= 0, x, x * 0.01)
   
   def derivative(self, x):
      return np.where(x >= 0, 1, 0.01)
   
class Elu:
   def __call__(self, x):
      return np.where(x > 0, x, np.exp(x) - 1)
   
   def derivative(self, x):
      return np.where(x > 0, 1, np.exp(x))

class Silu:
   def __call__(self, x):
      s = 1/(1+np.exp(-x))
      return x * s
   
   def derivative(self, x):
      s = 1/(1+np.exp(-x))
      return s + x * s * (1 - s)
   
def softmax(x):
   e_x = np.exp(x - np.max(x))
   return e_x / np.sum(e_x)


class NeuralNetwork:
    
  def __init__(self, activation):
    
    self.training_data = idx2numpy.convert_from_file(
        os.path.join(data_dir, 'train-images.idx3-ubyte')
    ).astype('float32') / 255.0
    self.training_data = np.transpose(self.training_data.reshape(self.training_data.shape[0], -1))
    self.testing_data = idx2numpy.convert_from_file(
        os.path.join(data_dir, 't10k-images.idx3-ubyte')
    ).astype('float32') / 255.0
    self.testing_data = np.transpose(self.testing_data.reshape(self.testing_data.shape[0], -1))

    self.training_labels = idx2numpy.convert_from_file(
        os.path.join(data_dir, 'train-labels.idx1-ubyte')
    )
    self.testing_labels = idx2numpy.convert_from_file(
        os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    )
    
    self.layers = []
    self.activations = []

    self.weights = []
    self.biases = []

    self.activate = activation
    self.derivative = activation.derivative

    self.learning_rate = 0.01

  def initialise_hidden_layers(self):
    num_layers = int(input('Enter number of hidden layers: '))
    for i in range(num_layers):
      self.layers.append(int(input(f'Enter number of neurons for hidden layer {i+1}: ')))

  def initialise_weights_and_biases(self):
      self.weights = [np.random.randn(self.layers[0], 784)]
      for i in range(len(self.layers)-1):
          self.weights.append(np.random.randn(self.layers[i+1],self.layers[i]))
      self.weights.append(np.random.randn(10, self.layers[-1]))

      for i in self.layers:
          self.biases.append(np.random.randn(i))
      self.biases.append(np.random.randn(10))
  

  def forwardPass(self, pixels):
      a = [pixels]
      pre_activation = []
      for i in range(len(self.layers)):
        z = np.dot(self.weights[i], a[i]) + self.biases[i]
        pre_activation.append(z)
        a.append(self.activate(z))

      z = np.dot(self.weights[-1], a[-1]) + self.biases[-1]
      pre_activation.append(z)
      a.append(softmax(z))
      
      prediction = np.argmax(a[-1])
      return prediction, a, pre_activation


  def backprop(self, y, a, pre_activation):
    weight_step = []
    bias_step = []

#    delta = 2 * (a[-1] - y) * a[-1] * (1 - a[-1])
    delta = 2 * (a[-1] - y)
    weight_step_1 = np.dot(delta.reshape(10, 1), a[-2].reshape(1, -1))
    weight_step.insert(0, weight_step_1)

    bias_step.insert(0, delta)

    for i in range(len(self.layers)):
      z_index = -(2 + i)
#      delta = np.dot(np.transpose(self.weights[z_index + 1]), delta) * a[z_index] * (1 - a[z_index])
#      delta = np.dot(np.transpose(self.weights[z_index + 1], delta)) * self.sigmoid_derivative(a[z_index])
      delta = np.dot(self.weights[z_index + 1].T, delta) * self.derivative(pre_activation[z_index])

      weight_step.insert(0, np.dot(delta.reshape(-1, 1), a[z_index - 1].reshape(1, -1)))
      bias_step.insert(0, delta)
    
    for i in range(len(self.weights)):
      self.weights[i] -= self.learning_rate * weight_step[i]
      self.biases[i] -= self.learning_rate * bias_step[i]

  def shuffle_training_data(self):
      samples = np.transpose(self.training_data)
      labels = self.training_labels
      
      combined = list(zip(samples, labels))
      np.random.shuffle(combined)
      shuffled_samples, shuffled_labels = zip(*combined)

      self.training_data = np.transpose(np.array(shuffled_samples))
      self.training_labels = np.array(shuffled_labels)
  
  def gradientDescent(self):
      self.shuffle_training_data()

      for i in range(self.training_data.shape[1]):
          training_drawing = self.training_data[:, i]
          result = self.forwardPass(training_drawing)
          y = np.zeros(10)
          y[self.training_labels[i]] = 1
          self.backprop(y, result[1], result[2])

  def testing(self):
      j = 0
      correct = 0
      count = 0
      for i in range(self.testing_data.shape[1]):
          test_drawing = self.testing_data[:, i]
          if self.forwardPass(test_drawing)[0] == self.testing_labels[i]:
              correct+= 1
          count += 1
          j += 1
          if j % 10 == 0:
            print(f'Prediction: {self.forwardPass(test_drawing)[0]} | Label: {self.testing_labels[i]}')
            print(correct / count)
            print('\n\n')


myneuralnet = NeuralNetwork(Sigmoid())

myneuralnet.initialise_hidden_layers()
myneuralnet.initialise_weights_and_biases()

num_epochs = int(input('Set number of epochs: '))
for epoch in range(num_epochs):
   print(f'Epoch {epoch + 1}')
   myneuralnet.gradientDescent()
  
myneuralnet.testing()

# This is my neural network
