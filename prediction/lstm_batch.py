import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.autograd import Variable
from analysis.analysis import get_sequence
import matplotlib.pyplot as plt
import math


class WorkloadPredictor:

    class LSTM(nn.Module):
        '''
            Parameter:
            - input_size : feature size
            - hidden_size : number of hidden units
            - output_size : number of output
            - num_layers: layers of LSTM of stack
        '''
        def __init__(self, input_size=1, hidden_size=128, output_size=1, num_layers=2):
            nn.Module.__init__(self)
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.dense = nn.Linear(in_features=hidden_size, out_features=output_size)
        def forward(self, size, h0, c0):
            _, (hn, cn) = self.lstm(size, (h0, c0))
            return self.dense(hn[-1])

    def __init__(self, data_len, h=100):
        self.h = h  # history length
        self.data_len = data_len
        self.batch_size = data_len - h  # batch size
        self.device = torch.device('cpu')
        if torch.cuda.is_available():  # check if GPU is available
            self.device = torch.device('cuda:0')
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')
        self.model = self.LSTM()
        self.h0 = torch.randn((2, self.batch_size, 128))
        self.c0 = torch.randn((2, self.batch_size, 128))

    def __train(self, train_x, train_y):
        train_x_tensor = torch.FloatTensor(train_x).unsqueeze(-1)
        train_y_tensor = torch.FloatTensor(train_y).unsqueeze(-1)
        print('train x tensor dimension:', Variable(train_x_tensor).size())
        criterion = F.mse_loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.9, last_epoch=-1)
        prev_loss = 1000
        max_epochs = 2000
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            output = self.model(train_x_tensor, self.h0, self.c0)
            loss = criterion(output, train_y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if loss < prev_loss:
                torch.save(self.model.state_dict(), 'lstm_model.pt')
                prev_loss = loss
            if loss.item() < 1e-4:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
                print('The loss value is reached')
                break
            elif (epoch + 1) % 100 == 0:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
        pred_y_for_train = self.model(train_x_tensor, self.h0, self.c0)
        pred_y_for_train = pred_y_for_train.view(-1, 1).data.numpy()
        return pred_y_for_train

    def __predict(self, test_x):
        pred_y = float(self.model(test_x, self.h0, self.c0)[-1][-1])
        return pred_y

    def __generate_train_data(self, data):
        train_x, train_y = [], []
        for i in range(self.batch_size):
            x = []
            for j in range(self.h):
                x.append(data[i + j])
            train_x.append(x)
            train_y.append(data[i + self.h])
        return train_x, train_y

    def __generate_predict_in(self, data):
        l = int(data.__len__() - (self.batch_size + self.h) + 1)
        data = data[l:]
        test_x = []
        for i in range(self.batch_size):
            x = []
            for j in range(self.h):
                x.append(data[i + j])
            test_x.append(x)
        test_x = torch.FloatTensor(test_x).unsqueeze(-1)
        return test_x

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
        min_val = min(seq)
        max_val = max(seq)
        interval = max_val-min_val
        def tsf(x):
            return (x-min_val)/interval
        def detsf(x):
            return x*interval + min_val
        for i in range(len(seq)):
            seq[i] = tsf(seq[i])
        train_data = seq[:self.data_len]
        # ----- train -----
        train_x, train_y = self.__generate_train_data(train_data)
        pred_y_for_train = self.__train(train_x, train_y)
        # ----- predict -----
        mape = 0
        num = 0
        t = []
        pred_y_for_test = []
        for ed in range(train_data.__len__(), seq.__len__() - 1):
            num += 1
            test_data = seq[:ed+1]
            test_x = self.__generate_predict_in(test_data)
            # print(test_x)
            pred_y = self.__predict(test_x)
            mape += abs(detsf(pred_y) - detsf(seq[ed+1])) / detsf(seq[ed+1])
            t.append(ed + 1)
            pred_y_for_test.append(detsf(pred_y))
        mape /= num
        print('mean absolute percentage error: ', mape)
        # ----- draw picture -----
        print('pred y for train:', pred_y_for_train)
        print('pred y for test:', pred_y_for_test)
        for i in range(len(seq)):
            seq[i] = detsf(seq[i])
        for i in range(len(pred_y_for_train)):
            pred_y_for_train[i] = detsf(pred_y_for_train[i])
        lines = [
            (range(len(seq)), seq, 'b', 'real_line'),
            (range(self.h, self.data_len), pred_y_for_train, 'y--', 'pred_train'),
            (t, pred_y_for_test, 'm--', 'pred_test')

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
    seq = get_sequence(path, window_size=1800)  # unit: minutes
    history_len = 16
    train_data_ratio = 0.7
    data_len = int(len(seq)*train_data_ratio)
    predictor = WorkloadPredictor(data_len=data_len, h=history_len)
    predictor.run(seq, name=trace_name)


if __name__ == '__main__':
    test()