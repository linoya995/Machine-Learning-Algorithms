import numpy as np


class Perceptron:
    """
    perceptron algorithm.
    """
    def __init__(self):
        self.w_perceptron = 0

    def training(self, x_train, y_train, label_set):
        # Set random seed (for reproducibility)
        np.random.seed(1500)
        d = len(x_train[0].split(",")) + 1  # Number of features
        m = len(x_train)  # Number of examples
        w = np.zeros((len(label_set), d))  # Weight vector

        epochs = 10
        eta = 0.1

        for t in range(0, epochs):
            s = list(zip(x_train, y_train))
            np.random.shuffle(s)

            for xt, yt in s:
                # Predict
                xt = np.fromstring(xt, dtype=float, sep=',')
                xt = np.append(xt, 1)  # adding Bias
                yt = int(float(yt))
                y_hat = np.argmax(np.dot(w, xt))

                # Update
                if yt != y_hat:
                    w[yt, :] = w[yt, :] + eta * xt
                    w[y_hat, :] = w[y_hat, :] - eta * xt
            eta = eta / (t + 1)

        # save wights
        self.w_perceptron = w

    def predict(self, x):
        """
        prediction of module for x
        :param x: the test
        :return: the prediction
        """
        x = np.fromstring(x, dtype=float, sep=',')
        x = np.append(x, 1)  # adding Bias
        y_hat = np.argmax(np.dot(self.w_perceptron, x))
        return y_hat
