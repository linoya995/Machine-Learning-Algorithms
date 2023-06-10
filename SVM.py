import numpy as np

class SVM:
    """
    SVM algorithm.
    """

    def __init__(self):
        self.w_svm = 0

    def training(self, x_train, y_train, label_set):
        # Set random seed (for reproducibility)
        np.random.seed(1500)
        m = len(x_train)  # Number of examples
        d = len(x_train[0].split(","))  # Number of features
        w = np.zeros((len(label_set), d))  # Weight matrix

        # Implement a SVM Classification
        epochs = 80
        eta = 0.02
        lamda = 0.001

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
                    w[yt] = ((1 - eta * lamda) * w[yt]) + eta * xt
                    w[y_hat] = ((1 - eta * lamda) * w[y_hat]) - eta * xt

                    for j in range(0, len(w)):
                        if j != yt and j != y_hat:
                            w[j] = ((1 - eta * lamda) * w[j])
            eta = eta / (i + 1)

        # save wights
        self.w_svm = w

    def predict(self, x):
        """
        prediction of module for x
        :param x: the test
        :return: the prediction
        """
        x = np.fromstring(x, dtype=float, sep=',')
        y_hat = np.argmax(np.dot(self.w_svm, x))
        return y_hat
