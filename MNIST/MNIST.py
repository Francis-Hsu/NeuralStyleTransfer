import time
import numpy as np
from chainer import datasets

train, test = datasets.get_mnist()

eta = 0.01  # learning rate
K = 10  # number of classes
L = 100  # number of hidden units
d = 784  # dimension of input


def softmax(z):
    ez = np.exp(z)
    return ez / np.sum(ez)


def h_sigmoid(z):
    ez = np.exp(z)
    return ez / np.add(ez, 1.0)


def h_relu(z):
    return np.maximum(z, 0)


def h_sigmoid_derivative(z):
    ez = np.exp(z)
    return ez / (np.add(ez, 1.0) ** 2.0)


def h_relu_derivative(z):
    return z > 0


def forward(x, w1, w2, b1, b2):
    z1 = np.dot(w1, x) + b1
    a2 = h_relu(z1)
    a2prime = h_relu_derivative(z1)
    z2 = np.dot(w2, a2) + b2
    f = softmax(z2)

    return f, a2, a2prime


def backward(x, y, w2, f, a, a_prime):
    delta3 = np.equal(range(10), y) - f
    delta2 = a_prime * np.dot(w2.transpose(), delta3)

    l_w1 = np.dot(delta2[:, None], x[None, :])
    l_w2 = np.dot(delta3[:, None], a[None, :])
    l_b1 = delta2
    l_b2 = delta3

    return l_w1, l_w2, l_b1, l_b2


def training(w1, w2, b1, b2):
    for i in np.random.permutation(len(train)):
        x = train[i][0]
        y = train[i][1]

        f, a, a_prime = forward(x, w1, w2, b1, b2)
        l_w1, l_w2, l_b1, l_b2 = backward(x, y, w2, f, a, a_prime)

        w1 += eta * l_w1
        w2 += eta * l_w2
        b1 += eta * l_b1
        b2 += eta * l_b2

    return w1, w2, b1, b2


def testing(w1, w2, b1, b2):
    num_correct = 0

    for i in range(len(test)):
        x = test[i][0]
        y = test[i][1]

        z1 = np.dot(w1, x) + b1
        a2 = h_relu(z1)
        z2 = np.dot(w2, a2) + b2
        f = softmax(z2)

        y_hat = f.argmax()
        num_correct += y == y_hat

    return num_correct / len(test)


def main():
    w1 = np.divide(np.random.rand(L, d), d)
    w2 = np.random.rand(K, L)
    b1 = np.random.rand(L)
    b2 = np.random.rand(K)

    time1 = time.time()
    for epoch in range(100):
        w1, w2, b1, b2 = training(w1, w2, b1, b2)
        accuracy = testing(w1, w2, b1, b2)
        print("Time Passed: %f, Epoch: %d, Test Accuracy: %.2f%%;"
              % (time.time() - time1, epoch + 1.0, accuracy * 100.0))


if __name__ == "__main__":
    main()
