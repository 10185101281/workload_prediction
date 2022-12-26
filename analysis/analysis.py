import csv
import time
import matplotlib.pyplot as plt
from datetime import datetime


def read_data(path):
    dividend = 1e7
    data = {}
    max_time = -1
    min_time = -1
    start_time = datetime.now()
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            timestamp = float(row[0]) // dividend # unit: s
            if max_time == -1:
                max_time = timestamp
                min_time = timestamp
            else:
                max_time = max(max_time, timestamp)
                min_time = min(min_time, timestamp)
            if timestamp in data.keys():
                data[timestamp] += 1
            else:
                data[timestamp] = 1
    end_time = datetime.now()
    interval_s = max_time-min_time
    interval_h = interval_s // 60 // 60
    print('[read data finish] used time:' + str((end_time-start_time).seconds) + '(s)')
    print('time interval: ' + str(interval_h) + '(h)')
    return data


def get_sequence(path, window_size=60*60):
    data = read_data(path)
    start_time = datetime.now()
    data_list = sorted(data.items(), key=lambda x:x[0])
    now = data_list[0][0]
    cnt = 0
    seq = []
    for time, num in data_list:
        while now + window_size <= time:
            seq.append(cnt)
            cnt = 0
            now += window_size
        cnt += num
    if cnt > 0:
        seq.append(cnt)
    end_time = datetime.now()
    print('[get sequence finish] used time:' + str((end_time-start_time).seconds) + '(s)' )
    return seq


def draw(lines, xlabel, ylabel, figsize=(16,8)):
    plt.figure(figsize=figsize)
    for line in lines:
        plt.plot(line[0], line[1], color='b', label=line[2])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.show()


def analysis():
    data_dir = r'/Users/baoliang/workspace/workload_prediction/data/msrc/MSR-Cambridge'
    name_list = [
        'hm_0.csv', 'hm_1.csv',
        'mds_0.csv', 'mds_1.csv',
        'prn_0.csv', 'prn_1.csv',
        'proj_0.csv', 'proj_1.csv', 'proj_2.csv', 'proj_3.csv', 'proj_4.csv',
        'prxy_0.csv', 'prxy_1.csv',

    ]
    for trace_name in name_list:
        path = data_dir + '//' + trace_name
        seq = get_sequence(path, window_size=60)
        line = (range(len(seq)), seq, trace_name)
        draw([line], 'time (min)', 'number of io')
        break


if __name__ == '__main__':
    analysis()
