import os
import math
import argparse
import numpy as np
from utils.gen_armiq_dataset import load_armiq_dataset,gen_artificial_anomaly_from_historgram
import matplotlib.pyplot as plt
import random
import pickle

def event_histogram(num_cls_event,_samples):
    """
    :param num_cls_event: number of events (event class)
    :param _samples: number of event samples
    :return: histogram: event historgram
    """
    histogram = np.zeros([num_cls_event],dtype=np.float32)
    for _data in _samples:
        histogram[_data] +=1.0
    return histogram


if __name__ == "__main__":
    try:
        _samples,_1,_2,_3 = load_armiq_dataset()
        histogram = event_histogram(150,_samples)
        xaxis = np.arange(150)
        _test = gen_artificial_anomaly_from_historgram(histogram,30,_2,_3)

        with open('anomaly.pickle','wb') as f:
            pickle.dump(_test,f)
            f.close()

        plt.bar(xaxis,histogram,width=1.0,color='r')
        plt.show()


    except KeyboardInterrupt as e:
        print("[STOP]", e)