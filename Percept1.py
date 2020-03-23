import numpy as np

def hardlim(x):
    if x < 0:
        return 0
    else:
        return 1


def perceptron(input, target):
    N, n = input.shape
    lr = 0.01
    w = np.random.randn(n, 1)
    E = 1

    while E > 0:
        E = 0
        for i in range(N):
            yi = hardlim(np.dot(input[i], w))
            ei = target[i] - yi
            w = w + lr * ei * input[i].reshape(n, 1)
            E = E + ei ^ 2


def fun_main():
    input_data = np.array([[1, 0], [0, 1], [1, 1]])
    target_data = np.array([0, 0, 0, 1])

    perceptron(input_data, target_data)

    print("AND({}, {}) -> {}".format(0, 0, perceptron(input_data, target_data)))
    print("AND({}, {}) -> {}".format(1, 0, perceptron(input_data, target_data)))
    print("AND({}, {}) -> {}".format(0, 1, perceptron(input_data, target_data)))
    print("AND({}, {}) -> {}".format(1, 1, perceptron(input_data, target_data)))

fun_main()
