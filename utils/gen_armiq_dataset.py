import numpy as np
import pickle
import glob
import pdb
import random


PATH = '/home/einstein/PycharmProjects/pythonProject/dataset/armiq_txt/'
DB_NAME = '../armiq_v0.txt'

def get_txtfile_list(path):
    myFiles = glob.glob(path+'*.txt')
    return myFiles


def length_norm(data,event_length,norm_length):
    """
    :param data: input datasety
    :param max_length: max length of events
    :param length: norm_length (at least larger than max_length)
    :return:
    """
    _len = len(data)
    _norm_size = np.zeros([_len,norm_length],dtype=np.float64)
    _mask = np.zeros([_len,norm_length],dtype=np.float64)
    for i,_tmp in enumerate(data):
        _norm_size[i,0:event_length[i]] = _tmp
        _mask[i,0:event_length[i]] = 1.0
    return _norm_size,_mask


def condition(x): return x ==0

def gen_artificial_anomaly_from_historgram(histogram,num_of_anomlay,min_length, max_length):
    event_logs_not_appeared = [idx for idx, element in enumerate(histogram) if condition(element)]
    abnormal_set = []
    _length_list = []
    for _i in range(num_of_anomlay):
        _length = random.randint(min_length,max_length)
        _length_list.append(_length)
        _sample= [random.choice(event_logs_not_appeared) for i in range(_length)]
        abnormal_set.append(_sample)
    return abnormal_set,_length_list


def load_armiq_dataset():
    _flist = get_txtfile_list(PATH)
    _set_list = []
    print('# of logfiles: %d'%(len(_flist)))
    _samples = [] #return value (dataset)
    with open(DB_NAME, 'w') as _dbf:
        _length_list = []
        for i,_fpath in enumerate(_flist):
            with open(_fpath) as _f:
                _log_list = []
                _int_list = []
                for line in _f:
                    _log_list.append(line.split('\t')[1].replace('\n',''))
                    _int_list.append(int(line.split('\t')[1].replace('\n','')))
                _tmp_length = len(_log_list)
                _length_list.append(_tmp_length)
                _dbf.writelines(','.join(_log_list))
                _samples.append(_int_list)
    print('The length of the longest even logs: %d'%(max(_length_list)))
    print('The length of the shortest even logs: %d'%(min(_length_list)))
    print('Avergae lenght of event logs: %f'%(np.mean(np.array(_length_list))))
    _dbf.close()
    return _samples,_length_list,min(_length_list),max(_length_list)


if __name__ == "__main__":
    try:
        load_armiq_dataset()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
