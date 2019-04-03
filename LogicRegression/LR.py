import numpy as np
from numpy import *
import math


class LR:
    def __init__(self):
        self.learning_rate = 0.1
        self.theta = ones((33, 1))
        self.prediction = []
        self.train_data = []
        self.load_date()

    def load_date(self):
        f = open('./data/trainSet.csv', 'r')
        f.readline()
        print('loading data...')
        for line in f.readlines():
            line_arr = line.strip('\n').split(',')
            line_arr = [float(i) for i in line_arr]
            line_arr.insert(0, 1)
            self.prediction.append(line_arr.pop())
            # print(line_arr)
            self.train_data.append(line_arr)
        f.close()
        self.train_data = np.array(self.train_data)

        index_max = np.max(self.train_data, axis=0)
        index_min = np.min(self.train_data, axis=0)
        print(index_max)
        print(index_min)

        for i in range(1, 33):
            for j in range(len(self.train_data)):
                self.train_data[j][i] = (self.train_data[j][i] - index_min[i])/(index_max[i] - index_min[i])

    def sigmoid(self, inX):
        return 1.0 / (1 + exp(-inX))

    def train(self):
        dataMat = mat(self.train_data)
        lableMat = mat(self.prediction)
        m,n = shape(dataMat)
        for i in range(10):
            h = self.sigmoid(dataMat*self.theta)
            error = lableMat - h
            self.theta = self.theta + self.learning_rate*dataMat.transpose()*error
            print('current accuracy: ', self.accuracy_train_set())

    def accuracy_train_set(self):
        result = []
        acc = 0
        for i in self.train_data:
            line = np.array(i)
            pre = np.sum(line*self.theta)
            if pre >= 0:
                result.append(1.0)
            else:
                result.append(0.0)
        print(result)
        for i in range(len(self.prediction)):
            if self.prediction[i] == result[i]:
                acc += 1
        return acc/len(self.prediction)


def main():
    lr = LR()
    lr.train()


if __name__ == '__main__':
    main()
