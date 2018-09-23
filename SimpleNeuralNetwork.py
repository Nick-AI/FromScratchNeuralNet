import gzip
import pickle
import numpy as np
from tqdm import tqdm
from urllib import request
from typing import List
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score


class FcLayer:

    def __init__(self, in_dim, nb_units, is_out_layer=False):
        self.is_out = is_out_layer
        if is_out_layer:
            self.weights = np.random.normal(0, np.sqrt(1/in_dim), (in_dim, nb_units))
        else:
            self.weights = np.random.normal(0, np.sqrt(2 / in_dim), (in_dim, nb_units))
        self.biases = np.zeros((1, nb_units))

    @staticmethod
    def relu(x, is_der=False):
        comp = np.zeros(x.shape)
        if is_der:
            x[x <= 0] = 0.
            x[x > 0] = 1.
            return x
        else:
            return np.max((x, comp), axis=0)

    @staticmethod
    def softmax(x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        out = exp / np.sum(exp, axis=1, keepdims=True)
        return out

    def get_output(self, inp):
        if self.is_out:
            out = np.dot(inp, self.weights) + self.biases
            self.out = self.softmax(out)
        else:
            out = np.dot(inp, self.weights) + self.biases
            self.out = self.relu(out)
        return self.out

    def get_gradient(self, follow_delta, prev_out):
        self.gradient = np.dot(prev_out.T, follow_delta)
        self.delta = np.dot(follow_delta, self.weights.T) * self.relu(prev_out, is_der=True)

    def update_params(self, prev_del, epsilon):
        self.weights += -epsilon * self.gradient
        self.biases += -epsilon * np.sum(prev_del, axis=0, keepdims=True)


class DataLoader:
    filename = [
        ["training_images", "train-images-idx3-ubyte.gz"],
        ["test_images", "t10k-images-idx3-ubyte.gz"],
        ["training_labels", "train-labels-idx1-ubyte.gz"],
        ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]

    def download_mnist(self):
        base_url = "http://yann.lecun.com/exdb/mnist/"
        for name in self.filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], name[1])
        print("Download complete.")

    def save_mnist(self):
        mnist = {}
        for name in self.filename[:2]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
        for name in self.filename[-2:]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.array(list(map(self.onehot_encode, np.frombuffer(f.read(), np.uint8, offset=8))))
        with open("./data/mnist.pkl", 'wb') as f:
            pickle.dump(mnist, f)
        print("Save complete.")

    @staticmethod
    def onehot_encode(y):
        out = np.zeros(10, dtype='float32')
        out[y-1] = 1.0
        return out

    def init(self):
        self.download_mnist()
        self.save_mnist()

    def load(self):
        with open("./data/mnist.pkl", 'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], \
               mnist["training_labels"], \
               mnist["test_images"], \
               mnist["test_labels"]


def construct_model(layer_dims, in_dims, out_dims):
    model = []
    for l_idx, l_dims in enumerate(layer_dims):
        if l_idx == 0:
            model.append(FcLayer(in_dims, l_dims))
        else:
            model.append(FcLayer(layer_dims[l_idx-1], l_dims))
    model.append(FcLayer(layer_dims[-1], out_dims, True))
    return model


def batch_gen(x, y, batch_size):
    for idx in range(0, len(x), batch_size):
        yield np.array(x[idx:idx+batch_size]), np.array(y[idx:idx+batch_size])


def train_model(model: List[FcLayer], train_x, train_y, val_x, val_y, batch_size, nb_epochs):
    train_loss = []
    test_loss = []
    epsilon = 0.001
    for ep in tqdm(range(nb_epochs)):
        if ep == int(0.6*nb_epochs):
            epsilon = epsilon*0.5
        if ep == int(0.8*nb_epochs):
            epsilon = epsilon*0.1
        loss = 0
        # Training
        for batch_x, batch_y in batch_gen(train_x, train_y, batch_size):
            predictions = []
            for l_idx, layer in enumerate(model):
                if l_idx == 0:
                    layer.get_output(batch_x)
                elif layer.is_out:
                    predictions = layer.get_output(model[l_idx-1].out)
                else:
                    layer.get_output(model[l_idx-1].out)
            error = (predictions - batch_y) / batch_size
            if loss != 0:
                loss = (loss + log_loss(batch_y, predictions))/2
            else:
                loss = log_loss(batch_y, predictions)
            for l_idx, layer in enumerate(model[::-1]):
                l_idx = len(model)-1-l_idx
                if l_idx == len(model)-1:  # if it is the last layer
                    layer.get_gradient(error, model[l_idx-1].out)
                    layer.update_params(error, epsilon)
                elif l_idx == 0:
                    layer.get_gradient(model[l_idx + 1].delta, batch_x)
                    layer.update_params(model[l_idx + 1].delta, epsilon)
                else:
                    layer.get_gradient(model[l_idx+1].delta, model[l_idx-1].out)
                    layer.update_params(model[l_idx+1].delta, epsilon)
        train_loss.append(loss)
        # Testing
        for l_idx, layer in enumerate(model):
            if l_idx == 0:
                layer.get_output(val_x)
            elif layer.is_out:
                predictions = layer.get_output(model[l_idx - 1].out)
            else:
                layer.get_output(model[l_idx - 1].out)
        loss = log_loss(val_y, predictions)
        test_loss.append(loss)
        if ep % 10 == 0:
            acc = accuracy_score(np.argmax(val_y, axis=1), np.argmax(predictions, axis=1))
            print('\nAccuracy:', acc)

    return train_loss, test_loss



def nn_driver():
    nb_classes = 10
    in_dims = 28 * 28
    loader = DataLoader()
    # loader.init()  # run only once
    train_x, train_y, test_x, test_y = loader.load()
    nn_model = construct_model(layer_dims=[100, 50, 20], in_dims=in_dims, out_dims=nb_classes)
    t_loss, v_loss = train_model(nn_model, train_x, train_y, test_x, test_y, batch_size=64, nb_epochs=100)
    plt.plot(t_loss, 'r', label='Training Loss')
    plt.plot(v_loss, 'b', label='Test Loss')
    plt.legend()
    plt.title('Loss Throughout Training')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    nn_driver()

