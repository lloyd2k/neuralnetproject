import numpy as np
import idx2numpy


training_data = idx2numpy.convert_from_file('train-images.idx3-ubyte').astype('float32') / 255.0
training_data = np.transpose(training_data.reshape(training_data.shape[0], -1))
testing_data = idx2numpy.convert_from_file('t10k-images.idx3-ubyte').astype('float32') / 255.0
testing_data = np.transpose(testing_data.reshape(testing_data.shape[0], -1))

training_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
testing_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

weights = [
    np.random.randn(16,784), # w[0]
    np.random.randn(16,16), # w[1]
    np.random.randn(10,16) # w[2]
]

biases = [
    np.random.randn(16), # b[0]
    np.random.randn(16), # b[1]
    np.random.randn(10) # b[2]
]

learning_rate = 0.01


def sigmoid(x):
     output = 1/(1+np.exp(-x))
     return output


def forwardPass(pixels):
    z1 = np.dot(weights[0], pixels) + biases[0]
    a1 = sigmoid(z1)

    z2 = np.dot(weights[1], a1) + biases[1]
    a2 = sigmoid(z2)

    z3 = np.dot(weights[2], a2) + biases[2]
    a3 = sigmoid(z3)

    prediction = np.argmax(a3)

    return a3, a2, a1, prediction


def backprop(pixels, y, a3, a2, a1):
    weight_step = [0,0,0]
    bias_step = [0,0,0]

    da3 = 2 * (a3 - y) * a3 * (np.ones_like(a3)- a3)
#   cost = (a3 - y) ** 2
#   cost_mean = np.mean(cost)
    weight_step[2] = np.dot(da3.reshape(10, 1),a2.reshape(1, -1))

    bias_step[2] = da3

    da2 = np.dot(np.transpose(weights[2]), da3)
    n2 = da2 * a2 * (np.ones_like(a2) - a2)
    weight_step[1] = np.dot(n2.reshape(16, 1), a1.reshape(1, -1))

    bias_step[1] = n2

    da1 = np.dot(np.transpose(weights[1]), da2)
    n1 = da1 * a1 * (np.ones_like(a1)- a1)
    weight_step[0] = np.dot(n1.reshape(16, 1), pixels.reshape(1, -1))

    bias_step[0] = n1

    weights[2] -= learning_rate * weight_step[2]
    weights[1] -= learning_rate * weight_step[1]
    weights[0] -= learning_rate * weight_step[0]

    biases[2] -= learning_rate * bias_step[2]
    biases[1] -= learning_rate * bias_step[1]
    biases[0] -= learning_rate * bias_step[0]


def gradientDescent():
    for i in range(training_data.shape[1]):
        training_drawing = training_data[:, i]
        forwardPass(training_drawing)
        y = np.zeros(10)
        y[training_labels[i]] = 1
        backprop(training_drawing, y, forwardPass(training_drawing)[0], forwardPass(training_drawing)[1], forwardPass(training_drawing)[2])


def testing():
    j = 0
    correct = 0
    count = 0
    for i in range(testing_data.shape[1]):
        test_drawing = testing_data[:, i]
        if forwardPass(test_drawing)[3] == testing_labels[i]:
            correct+= 1
        count += 1
        j += 1
        if j % 10 == 0:
           print(forwardPass(test_drawing)[3], testing_labels[i])
           print(correct / count)
           print('\n\n')


gradientDescent()
testing()
