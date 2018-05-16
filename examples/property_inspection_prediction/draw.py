#!/usr/bin/env python

import sys
import datetime
import matplotlib.pyplot as plt


def search(string, start, end):
    start_pos = string.find(start)
    if start_pos > -1:
        end_pos = string.find(end, start_pos)
        return string[(start_pos + len(start)):end_pos]
    else:
        return None


def discover(string, start, middle, end):
    start_pos = string.find(start)
    if start_pos > -1:
        mid_pos = string.find(middle, start_pos)
        if mid_pos > -1:
            end_pos = string.find(end, mid_pos)
            return string[(mid_pos + 2):end_pos]
    return None


def get_metrics(lines, metric_name):
    indices = []
    metrics = []
    for line in lines:
        metric = discover(line, metric_name, ':', '\n')
        if metric is not None:
            metrics.append(float(metric))
            idx = search(line, 'Iteration:', ',')
            indices.append(int(idx))
    return indices, metrics


if __name__ == '__main__':

    try:
        file_name = sys.argv[1]
    except IndexError:
        raise Exception('Provide the log file name.')

    try:
        with open(file_name, 'r') as f:
            lines = f.readlines()
    except Exception:
        raise Exception('Cannot read %s.' % file_name)

    train_indices, train_metrics = get_metrics(lines, 'training')
    valid_indices, valid_metrics = get_metrics(lines, 'valid_1')

    print('%s smallest metric on valid: %f' % (file_name[:2], min(valid_metrics)))

    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)
    ax.plot(train_indices, train_metrics, 'k-', linewidth=2, label='train')
    ax.plot(valid_indices, valid_metrics, 'r-', linewidth=2, label='valid')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Metrics')
    ax.legend(loc='best')
    ax.set_title(file_name[:2])
    ax.grid()
    # fig.savefig(
    #     'pic_%s.png' % datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    # )
    fig.savefig('%s_pic.png' % file_name[:2])
