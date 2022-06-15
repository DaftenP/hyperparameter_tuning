import numpy as np
import os
import random


class DataReader():
    def __init__(self):
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.read_data()

    def read_data(self):
        filename = os.listdir("datasets")[1]
        print(filename)
        file = open("datasets/" + filename)

        file.readline()

        data = []

        for line in file:
            splt = line.split(",")

            x, label = self.process_data(splt)

            data.append((x, label))

        random.shuffle(data)

        X = []
        Y = []

        for el in data:
            X.append(el[0])
            Y.append(el[1])

        X = np.asarray(X)
        Y = np.asarray(Y)

        train_X = X[:int(len(X) * 0.8)]
        train_Y = Y[:int(len(Y) * 0.8)]
        test_X = X[int(len(X) * 0.8):]
        test_Y = Y[int(len(Y) * 0.8):]

        return train_X, train_Y, test_X, test_Y

    def process_data(self, splt):
        data = []
        for i in range(1, len(splt)):
            if splt[i] == "": splt[i] = 0
        data = list(map(float, splt[1:-1]))

        label = list(map(int, splt[-1:]))

        return data, label
