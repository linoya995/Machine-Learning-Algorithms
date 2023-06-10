import sys
from Perceptron import *
from PassiveAggressive import *
from SVM import *


def main(argv):
    data_list = []

    # go over the files
    for fileName in argv:
        # Reading a Text File
        lines = [line.rstrip('\n') for line in open(fileName)]
        lines = [w.replace('M', '0') for w in lines]
        lines = [w.replace('F', '0.5') for w in lines]
        lines = [w.replace('I', '1') for w in lines]
        data_list.append(lines)

    x_train = data_list[0]  # Examples list
    y_train = data_list[1]  # Labels list
    label_set = set(y_train)  # All possible labels

    # train algorithms
    p = Perceptron()
    p.training(x_train, y_train, label_set)
    svm = SVM()
    svm.training(x_train, y_train, label_set)
    pa = PassiveAggressive()
    pa.training(x_train, y_train, label_set)

    # test file
    tests = data_list[2]

    for i in range(0, len(tests)):
        # get algorithms predictions
        perceptron_predict = p.predict(tests[i])
        svm_predict = svm.predict(tests[i])
        passive_aggressive_predict = pa.predict(tests[i])

        print("perceptron: " + str(perceptron_predict) + ", svm: " + str(svm_predict) + ", pa: " +
              str(passive_aggressive_predict))


if __name__ == '__main__':
    main(sys.argv[1:])
