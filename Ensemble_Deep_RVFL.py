#-- coding: utf-8 --
#@Time : 2021/3/27 20:40
#@Author : HUANG XUYANG
#@Email : xhuang032@e.ntu.edu.sg
#@File : Ensemble_Deep_RVFL.py
#@Software: PyCharm

import numpy as np
from sklearn.datasets import load_digits


num_nodes = 50  # Number of enhancement nodes.
regular_para = 0.05  # Regularization parameter.
random_range = [-1, 1]  # Range of random weights.
num_layer = 5  # Number of hidden layers


class EnsembleDeepRVFL:
    """A ensemble deep RVFL classifier.

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        random_range: A list, [min, max], the range of generating random weights or bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        n_layer: A integer, N=number of hidden layers.
    """
    def __init__(self, n_nodes, lam, random_vec_range, activation, n_layer):
        self.n_nodes = n_nodes
        self.lam = lam
        self.random_range = random_vec_range
        self.random_weights = []
        self.random_bias = []
        self.beta = []
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.n_layer = n_layer

    def train(self, data, label):
        """

        :param data: Training data.
        :param label: Training label.
        :return: No return
        """

        assert len(data.shape) > 1, f'data shape should be [n, dim].'
        assert len(data) == len(label), f'label number does not match data number.'

        data = self.normalize(data)  # Normalization data
        n_sample = len(data)
        n_feature = len(data[0])
        h = data
        y = self.one_hot(label, num_class)
        for i in range(self.n_layer):
            self.random_weights.append(self.get_random_vectors(len(h[0]), self.n_nodes, self.random_range))
            self.random_bias.append(self.get_random_vectors(1, self.n_nodes, self.random_range))
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)

            h = d

            if n_sample > (self.n_nodes + n_feature):
                self.beta.append(np.linalg.inv((self.lam * np.identity(n_feature + self.n_nodes) + np.dot(d.T, d))).dot(d.T).dot(y))
            else:
                self.beta.append(d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y))

    def predict(self, data, output_prob=False):
        """

        :param data: Predict data.
        :param output_prob: A bool number, if True return the raw predict probability, if False return predict class.
        :return: Prediction result.
        """
        data = self.normalize(data)  # Normalization data
        n_sample = len(data)
        h = data
        results = []
        for i in range(self.n_layer):
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)

            h = d
            if not output_prob:
                results.append(np.argmax(np.dot(d, self.beta[i]), axis=1))
            else:
                results.append(np.dot(d, self.beta[i]))
        if not output_prob:
            results = list(map(np.bincount, list(np.array(results).transpose())))
            results = np.array(list(map(np.argmax, results)))
        return results

    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: Accuracy.
        """
        data = self.normalize(data)  # Normalization data
        n_sample = len(data)
        h = data
        results = []
        for i in range(self.n_layer):
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]),
                                                                                    self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)

            h = d
            results.append(np.argmax(np.dot(d, self.beta[i]), axis=1))
        results = list(map(np.bincount, list(np.array(results).transpose())))
        results = np.array(list(map(np.argmax, results)))
        acc = np.sum(np.equal(results, label))/len(label)
        return acc

    def get_random_vectors(self, m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        return x

    def one_hot(self, x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def normalize(self, x):
        return (x - np.mean(x)) / np.std(x)


class Activation:
    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))

    def sine(self, x):
        return np.sin(x)

    def hardlim(self, x):
        return (np.sign(x) + 1) / 2

    def tribas(self, x):
        return np.maximum(1 - np.abs(x), 0)

    def radbas(self, x):
        return np.exp(-(x**2))

    def sign(self, x):
        return np.sign(x)

    def relu(self, x):
        return np.maximum(0, x)


def prepare_data(proportion):
    digits_dataset = load_digits()
    label = digits_dataset['target']
    data = digits_dataset['data']
    n_class = len(digits_dataset['target_names'])

    shuffle_index = np.arange(len(label))
    np.random.shuffle(shuffle_index)

    train_number = int(proportion * len(label))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]
    data_train = data[train_index]
    label_train = label[train_index]
    data_val = data[val_index]
    label_val = label[val_index]
    return (data_train, label_train), (data_val, label_val), n_class


if __name__ == '__main__':
    train, val, num_class = prepare_data(0.9)
    ensemble_deep_rvfl = EnsembleDeepRVFL(num_nodes, regular_para, random_range, 'relu', num_layer)
    ensemble_deep_rvfl.train(train[0], train[1])
    prediction = ensemble_deep_rvfl.predict(val[0], output_prob=False)
    accuracy = ensemble_deep_rvfl.eval(val[0], val[1])
