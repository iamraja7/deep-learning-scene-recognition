from datetime import datetime

from terminaltables import AsciiTable
import numpy as np

from layers.utils import iter_batch, get_time_diff


class NeuralNetwork:

    def __init__(self, opt_type, loss, val_datas=None):
        self.list_layers = []
        self.opt_type = opt_type
        self.loss_func = loss()
        self.er_dict = {"validation": [], "training": []}
        self.validset = None
        if val_datas:
            X, y = val_datas
            self.validset = {"X": X, "y": y}

    # Implementing the add function for the layers
    def add(self, layer):
        if self.list_layers:
            layer.set_input(shape=self.list_layers[-1].get_output())
        if hasattr(layer, 'initialize_value'):
            layer.initialize_value(optimizer=self.opt_type)
        self.list_layers.append(layer)

    # Function to calculate loss and accuracy for test data
    def test_batch(self, X, y):
        y_predict = self._front_pass(X, training=False)
        lossval = np.mean(self.loss_func.loss(y, y_predict))
        accuracy = self.loss_func.calculate_accuracy(y, y_predict)
        return lossval, accuracy

    # Function to calculate loss and accuracy for train data
    def train_batch(self, X, y):
        y_predict = self._front_pass(X)
        lossval = np.mean(self.loss_func.loss(y, y_predict))
        accuracy = self.loss_func.calculate_accuracy(y, y_predict)
        lossgradient = self.loss_func.gradient(y, y_predict)
        self._backward_pass(loss_gradient=lossgradient)

        return lossval, accuracy

    # Function to fit the data to the model
    def fit(self, X, y, nepochs, batch_size):
        train_acc = []
        val_acc = []
        total_start_time = datetime.now()
        for i in range(nepochs):
            batcherror = []
            batch_train_accuracy = []
            val_accuracy = 0
            batch = 1
            epoch_start_time = datetime.now()
            for Xbatch, ybatch in iter_batch(X, y, batch_size=batch_size):
                loss, train_accuracy = self.train_batch(Xbatch, ybatch)
                batcherror.append(loss)
                batch_train_accuracy.append(train_accuracy)
                print("Training for epoch:{} batch:{} in time:{} | loss={:.2f}, accuracy={:.2f}"
                      .format(i, batch, get_time_diff(epoch_start_time), loss, train_accuracy), end='\r')
                batch += 1
            print("")

            if self.validset is not None:
                valloss, val_accuracy = self.test_batch(self.validset["X"], self.validset["y"])
                self.er_dict["validation"].append(valloss)

            mean_trainingloss = np.mean(batcherror)
            mean_training_accuracy = np.mean(batch_train_accuracy)
            train_acc.append(mean_training_accuracy)
            val_acc.append(val_accuracy)

            self.er_dict["training"].append(mean_trainingloss)
            print(
                "Training loop complete for epoch:{} in time:{} | train_loss:{:.2f} train_accuracy:{:.2f} | val_loss:{:.2f} val_accuracy:{:.2f}"
                    .format(i, get_time_diff(epoch_start_time), mean_trainingloss, mean_training_accuracy, valloss,
                            val_accuracy))

        print("Final accuracy:{:.2f} | Time taken:{}".format(val_acc[-1], get_time_diff(total_start_time)))
        return self.er_dict["training"], self.er_dict["validation"], train_acc, val_acc

    # Defining forward pass
    def _front_pass(self, X, training=True):
        l_out = X
        for l in self.list_layers:
            l_out = l.front_flow(l_out, training)
        return l_out

    # Defining backward pass
    def _backward_pass(self, loss_gradient):
        for l in reversed(self.list_layers):
            loss_gradient = l.back_flow(loss_gradient)

    # Defining summary for the model
    def summary(self, name="Model Summary"):
        print(AsciiTable([[name]]).table)
        print("Input Shape: %s" % str(self.list_layers[0].inp_size))
        tab_val = [["Name of Layer", "Params", "Output Shape"]]
        total_parameters = 0
        for l in self.list_layers:
            l_name = l.name()
            parameters = l.params()
            output_shape = l.get_output()
            tab_val.append([l_name, str(parameters), str(output_shape)])
            total_parameters += parameters
        print(AsciiTable(tab_val).table)
        print("Total Parameters are: %d\n" % total_parameters)

    # Defining predict function
    def predict(self, X):
        return self._front_pass(X, training=False)
