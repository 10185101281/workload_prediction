import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from analysis.analysis import get_sequence
from torch.autograd import Variable
import numpy as np
import pandas as pd


class WorkloadPredictor:

    class LSTM(nn.Module):
        '''
            Parameter:
            - input_size : feature size
            - hidden_size : number of hidden units
            - output_size : number of output
            - num_layers: layers of LSTM of stack
        '''
        def __init__(self, input_size, hidden_size=128, output_size=1, num_layers=2):
            nn.Module.__init__(self)
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.dense = nn.Linear(in_features=hidden_size, out_features=output_size)
        def forward(self, _x):
            x , _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
            s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
            x = x.view(s*b, h)
            x = self.dense(x)
            x = x.view(s, b, -1)
            return x

    def __init__(self, h=100):
        self.h = h  # history length
        self.INPUT_FEATURE_NUM = self.h
        self.OUTPUT_FEATURE_NUM = 1

        self.device = torch.device('cpu')
        if torch.cuda.is_available():  # check if GPU is available
            self.device = torch.device('cuda:0')
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')

        self.model = self.LSTM(self.h)
        print('LSTM model:',self.model)
        print('model.parameters:',self.model.parameters)

    def __train(self, train_x, train_y):
        train_x_tensor = train_x.reshape(-1, 1, self.INPUT_FEATURE_NUM)
        train_y_tensor = train_y.reshape(-1, 1, self.OUTPUT_FEATURE_NUM)
        train_x_tensor = torch.from_numpy(train_x_tensor)
        train_y_tensor = torch.from_numpy(train_y_tensor)
        print('train x tensor dimension:', Variable(train_x_tensor).size())
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.9, last_epoch=-1)
        prev_loss = 1000
        max_epochs = 2000
        train_x_tensor = train_x_tensor.to(self.device)
        for epoch in range(max_epochs):
            output = self.model(train_x_tensor).to(self.device)
            loss = criterion(output, train_y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss < prev_loss:
                torch.save(self.model.state_dict(), 'lstm_model.pt')
                prev_loss = loss
            if loss.item() < 1e-4:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
                print('The loss value is reached')
            elif (epoch + 1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
        pred_y_for_train = self.model(train_x_tensor).to(self.device)
        pred_y_for_train = pred_y_for_train.view(-1, self.OUTPUT_FEATURE_NUM).data.numpy()
        return pred_y_for_train

    def __predict(self, test_x, test_y):
        self.model = self.model.eval()  # switch to testing model
        test_x_tensor = test_x.reshape(-1, 1, self.INPUT_FEATURE_NUM)
        test_x_tensor = torch.from_numpy(test_x_tensor)
        test_x_tensor = test_x_tensor.to(self.device)
        pred_y_for_test = self.model(test_x_tensor).to(self.device)
        pred_y_for_test = pred_y_for_test.view(-1, self.OUTPUT_FEATURE_NUM).data.numpy()
        criterion = nn.MSELoss()
        loss = criterion(torch.from_numpy(pred_y_for_test), torch.from_numpy(test_y))
        print('test loss: ', loss)
        return pred_y_for_test

    def __get_x_y(self, seq):
        seq_len = len(seq)
        data_x = []
        data_y = []
        for i in range(seq_len - self.h):
            x = []
            for j in range(self.h):
                x.append(seq[i + j])
            y = seq[i + self.h]
            data_x.append(x)
            data_y.append(y)
        return data_x, data_y

    def __draw(self, lines, xlabel, ylabel, title, figsize=(16, 8)):
        plt.figure(figsize=figsize)
        plt.title(title)
        for line in lines:
            plt.plot(line[0], line[1], line[2], label=line[3])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.show()

    def run(self, seq, name):
        data_x, data_y = self.__get_x_y(seq)
        data_x = np.array(data_x).astype('float32')
        data_y = np.array(data_y).astype('float32')
        data_len = len(data_x)
        train_data_ratio = 0.7
        train_data_len = int(data_len * train_data_ratio)

        train_x = data_x[:train_data_len]
        train_y = data_y[:train_data_len]
        test_x = data_x[train_data_len:]
        test_y = data_y[train_data_len:]
        # ----- train -----
        pred_y_for_train = self.__train(train_x, train_y)
        # ----- predict -----
        pred_y_for_test = self.__predict(test_x, test_y)
        # ----- draw picture -----
        lines = [
            (range(len(seq)), seq, 'b', 'real_line'),
            (range(self.h, self.h+train_data_len), pred_y_for_train, 'y--', 'pred_train'),
            (range(self.h+train_data_len, self.h+data_len), pred_y_for_test, 'm--', 'pred_test')

        ]
        self.__draw(lines, xlabel='time (min)', ylabel='number of io', title=name)


def test():
    data_dir = r'/Users/baoliang/workspace/workload_prediction/data/msrc/MSR-Cambridge'
    name_list = [
        'hm_0.csv', 'hm_1.csv',
        'mds_0.csv', 'mds_1.csv',
        'prn_0.csv', 'prn_1.csv',
        'proj_0.csv', 'proj_1.csv', 'proj_2.csv', 'proj_3.csv', 'proj_4.csv',
        'prxy_0.csv', 'prxy_1.csv',

    ]
    trace_name = name_list[0]
    path = data_dir + '//' + trace_name
    seq = get_sequence(path, window_size=3600)  # unit: minutes
    history_len = 32
    predictor = WorkloadPredictor(h=history_len)
    predictor.run(seq, name=trace_name)


if __name__ == '__main__':
    test()