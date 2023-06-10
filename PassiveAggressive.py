import numpy as np


class PassiveAggressive:
    """
    PassiveAggressive algorithm.
    """

    def __init__(self):
        self.w_passive_aggressive = 0

    def training(self, x_train, y_train, label_set):
        # Set random seed (for reproducibility)
        np.random.seed(1500)
        m = len(x_train)  # Number of example
        d = len(x_train[0].split(","))  # Number of features
        w = np.zeros((len(label_set), d))  # Weight matrix

        # Implement a Passive Aggressive Classification
        epochs = 1000
        for i in range(0, epochs):
            s = list(zip(x_train, y_train))
            np.random.shuffle(s)

            for xt, yt in s:
                # Predict
                xt = np.fromstring(xt, dtype=float, sep=',')
                yt = int(float(yt))
                y_hat = np.argmax(np.dot(w, xt))

                # Update
                if yt != y_hat:
                    r = 1 - (np.dot(w[yt], xt) + np.dot(w[y_hat], xt))
                    loss = max(0, r)
                    tau = loss / (2 * (np.power(np.linalg.norm(xt, ord=2), 2)))
                    w[yt] = w[yt] + tau * xt
                    w[y_hat] = w[y_hat] - tau * xt

        # save wights
        self.w_passive_aggressive = w

    def predict(self, x):
        """
        prediction of module for x
        :param x: the test
        :return: the prediction
        """
        x = np.fromstring(x, dtype=float, sep=',')
        y_hat = np.argmax(np.dot(self.w_passive_aggressive, x))
        return y_hat
