import numpy as np
import pickle
import glob
import pdb
import random
import os
import pathlib


PATH1 = '/home/einstein/PycharmProjects/pythonProject/dataset/armiq_txt_seq/'
PATH2 = '/home/einstein/PycharmProjects/pythonProject/dataset/armiq_txt_cosmetic/'
DB_NAME = '../armiq_v0.txt'


def get_list_of_files_in_dir(PATH):
    _list= []
    for root, dirs, files in os.walk(PATH):
        for file in files:
            if file.endswith(".txt"):
                _list.append(os.path.join(root, file))
    return _list

    return [f for f in pathlib.Path(directory).glob(file_types) if f.is_file()]

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
        _norm_size[i,0:len(_tmp)] = _tmp
        _mask[i,0:len(_tmp)] = 1.0
    return _norm_size,_mask


def condition(x): return x ==0


def gen_anomlay_with_length(abn_list,norm_list,length,nor_portion=0.3):
    if nor_portion == -1:
        nor_portion = random.random()
        if nor_portion > 0.01:
            nor_portion = 1
    _len_ab = int(length*(1-nor_portion))
    _len_norm = int(length*(nor_portion))
    ab_part = [random.choice(abn_list) for i in range(_len_ab)]
    norm_part = [random.choice(norm_list) for i in range(_len_norm)]
    sample = np.concatenate((ab_part,norm_part))
    np.random.shuffle(sample)
    return sample

def gen_artificial_anomaly_from_historgram(histogram,num_of_anomlay,min_length, max_length,_normal_portion=0.3):
    event_logs_not_appeared = [idx for idx, element in enumerate(histogram) if condition(element)]
    event_logs_appeared = [idx for idx, element in enumerate(histogram) if condition(element)==False]
    abnormal_set = []
    _length_list = []
    for _i in range(num_of_anomlay):
        _length = random.randint(min_length+2,max_length)
        _length_list.append(_length)
        _sample= gen_anomlay_with_length(event_logs_not_appeared,event_logs_appeared,_length,_normal_portion)
        abnormal_set.append(_sample)
    return abnormal_set,_length_list


def load_armiq_dataset():
    _flist1 = get_list_of_files_in_dir(PATH1)
    _flist2 = get_list_of_files_in_dir(PATH2)
    _flist = _flist1+_flist2
    _set_list = []
    print('# of logfiles: %d'%(len(_flist)))
    _samples = [] #return value (dataset)
    _longest_file = ''
    _longest_log = 0
    _invalid_to_long = 0
    _valid = 0
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

                if len(_log_list) > _longest_log:
                    _longest_log = len(_log_list)
                    _longest_file = _fpath
                if len(_log_list) > 4000:
                    _invalid_to_long +=1
                    continue
                else:
                    _valid +=1
                    _length_list.append(_tmp_length)
                    _dbf.writelines(','.join(_log_list))
                    _samples.append(_int_list)
    print('The length of the longest even logs: %d (%s)'%(max(_length_list),_longest_file))
    print('The length of the shortest even logs: %d (%s)'%(min(_length_list),_longest_file))
    print('Avergae lenght of event logs: %f (%s)'%(np.mean(np.array(_length_list)),_longest_file))
    print('Invalid sample :%d (%.3f), valid samples: %d (%.3f)'%(_invalid_to_long,_invalid_to_long/(_invalid_to_long+_valid),_valid,_valid/(_invalid_to_long+_valid)))
    _dbf.close()
    return _samples,_length_list,min(_length_list),max(_length_list)


if __name__ == "__main__":
    try:
        load_armiq_dataset()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
