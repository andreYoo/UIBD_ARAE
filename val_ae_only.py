import os
import math
import argparse
from sklearn.utils import shuffle
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.nn import functional as F
from models.Simple_Nets import Encoder, Decoder, RVAE,Latent_Discriminator
from torch.autograd import Variable
from utils.utils import load_dataset
from utils.gen_armiq_dataset import load_armiq_dataset, length_norm, gen_artificial_anomaly_from_historgram
from utils.armiq_data_loader import ARMIQ_TRAIN_DATALOADER, ARMIQ_EVAL_DATALOADER
from tensorboardX import SummaryWriter
from MulticoreTSNE import MulticoreTSNE as TSNE
import dataset_analysis as da
import numpy as np
import pandas as pd
import csv

import pdb
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
N = 150
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)



_tv_ratio = 0.9941 #Train versus validation ratio
val_num = 500
BATCH_SIZE = 32
_lambda = 0.01
_global_step = 0
check_interval = 100

summary = SummaryWriter('./logs/latent_ae')



def scatter_hist(x, y, ax, ax_histx, ax_histy,label,color):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    #ax.scatter(x, y,c=label,cmap=color,s=0.5)
    ax.scatter(x, y,c='black',s=0.5)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins,color='black')
    ax_histy.hist(y, bins=bins, orientation='horizontal',color='black')

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=30,help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()

def sample_wise_reconstruction_error(output,annot,mask):
    _shape = np.shape(mask)
    mask = mask.flatten().float()
    error = (output-annot)**2
    error = torch.mul(mask.view(-1,1),error.view(_shape[0]*_shape[1],-1))
    error = error.view(_shape[0],_shape[1],-1)
    return error.sum(1).sum(1)

def loss_masked(output,annot,mask,type='train'):
    _shape = np.shape(output)
    mask = mask.flatten().float()
    dif = output-annot
    dif = dif.view([_shape[0]*_shape[1],_shape[2]])
    return torch.sum(torch.matmul(mask,((dif) ** 2)))


def validation(model, val_loader, epoch,recon=False):
    model = model.cpu()
    total_loss = 0
    _tmp_loss = 0.0
    _len = len(val_loader)
    for b, batch in enumerate(val_loader):
        print("%d/%d"%(b,len(val_loader)))
        src = batch['event'].long()
        msk = batch['mask'].float()
        msk = msk.cpu()
        src = src.cpu()

        output, trg, latent = model(src)
        loss = loss_masked(output, trg, msk)
        msk = msk.cpu()
        latent = latent.cpu()
        src = src.cpu()
        _shape = np.shape(latent)
        src_fatten = src.view([-1])
        msk_flatten = msk.view([-1])
        latent = latent.view([_shape[0] * _shape[1], _shape[2]])
        latent = latent[msk_flatten == 1, :]

        src_flatten = src_fatten[msk_flatten == 1]

        total_loss += loss.data.item()
        if b==0:
            _latent_list = latent.detach().numpy()
            _event_code_list = src_flatten.detach().numpy()
        else:
            _latent_list = np.concatenate((_latent_list,latent.detach().numpy()),axis=0)
            _event_code_list = np.concatenate((_event_code_list,src_flatten.detach().numpy()),axis=0)
    print('[Validation][%d epoch %d step] - Total loss: %.3f ' % (epoch, b, total_loss / _len))
    #pdb.set_trace()
    tsne_model = TSNE(learning_rate=100,n_jobs=6)
    print('Dim reduction - T-SNE - ing')
    tf_latent = tsne_model.fit_transform(_latent_list)
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    tf_latent = tf_latent[_event_code_list==67]

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.1]
    rect_histy = [left + width + spacing, bottom, 0.1, height]
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    if recon==True:
        ax.set_ylabel('Reconstruction loss only', fontsize=30)
    else:
        ax.set_ylabel('Total loss (Recon loss + Adv loss)', fontsize=30)
    # ax.set_xticks([])
    # ax.set_yticks([])

    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # ax_histx.set_xticks([])
    # ax_histx.set_yticks([])
    #
    # ax_histy.set_xticks([])
    # ax_histy.set_yticks([])
    ax.grid()


    # use the previously defined function
    scatter_hist(tf_latent[:,0], tf_latent[:,1], ax, ax_histx, ax_histy,_event_code_list,cmap)
    fig.suptitle('%d Epoch'%(epoch), fontsize=40,y=0.9)
    #plt.scatter(tf_latent[:,0], tf_latent[:,1], c='b',s=0.5)
    if recon==True:
        plt.savefig('./plot_dist/%d-epoch_recon.pdf'%(epoch))
    else:
        plt.savefig('./plot_dist/%d-epoch_total.pdf'%(epoch))
    return total_loss / _len

#Evaluation
def evaluate(model,data_loader):
    for b, batch in enumerate(data_loader):
        src = batch['event'].long().cuda()
        msk = batch['mask'].float().cuda()
        annot = batch['annot'].float().cuda()
        output, trg, latent = model(src)
        _shape = np.shape(latent)
        msk_flatten=msk.view([-1])
        latent = latent.view([_shape[0]*_shape[1],_shape[2]])
        latent= latent[msk_flatten==1,:]
        recon_error = torch.log10(sample_wise_reconstruction_error(output,trg,msk))
        if b ==0:
            event_list = src.data.cpu()
            recon_error_list = recon_error.data.cpu()
            annot_list = annot.data.cpu()
        else:
            event_list = np.concatenate((event_list,src.data.cpu()))
            recon_error_list = np.concatenate((recon_error_list,recon_error.data.cpu()))
            annot_list = np.concatenate((annot_list,annot.data.cpu()))
    return event_list,recon_error_list,annot_list


def train(model, optimizer, train_loader, grad_clip,epoch):
    model = model.cuda()
    model.train()
    total_loss = 0
    _tmp_loss = 0.0
    _tmp_re_loss = 0.0
    _tmp_adv_loss = 0.0
    for b, batch in enumerate(train_loader):
        _global_step
        src = batch['event'].long()
        msk = batch['mask'].float()
        #src, len_src = torch.from_numpy(np.array(batch)).view(1,len(batch)), len(batch)
        #src, len_src = src.view(1,len(batch)), len(src)np.
        msk = Variable(msk.cuda())
        src = src.cuda()


        output,trg, latent = model(src)
        _shape = np.shape(latent)
        msk_flatten=msk.view([-1])
        latent = latent.view([_shape[0]*_shape[1],_shape[2]])
        latent= latent[msk_flatten==1,:]

        _filted_len = len(latent)

        optimizer.zero_grad()
        loss = loss_masked(output,trg,msk)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        step_loss=  loss.data.item()

        _tmp_loss += step_loss


        total_loss += loss.data.item()
        summary.add_scalar('loss/loss',step_loss,epoch * len(train_loader) + b)
        if b%check_interval==0:
            print('[training][%d epoch %d step] - Total loss: %.3f '%(epoch,b,_tmp_loss/check_interval))
            _tmp_loss = 0.0
    return total_loss

def main():
    args = parse_arguments()
    hidden_size = 256
    embed_size = 256
    log_numbers = 150
    hidden_size_dec = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")

    #Load dataset
    #pdb.set_trace()
    train_set, length_list, min_length, max_length = load_armiq_dataset()
    _train_set_histogram = da.event_histogram(150, train_set)
    abnormal_set, ab_length_list = gen_artificial_anomaly_from_historgram(_train_set_histogram,
                                                                          int(len(train_set) * 0.3), min_length,
                                                                          max_length)

    tmp_set, tmp_mask = length_norm(train_set, length_list, max_length)

    _train_lenth = len(train_set)
    # Shuffle
    tmp_set, tmp_mask = shuffle(tmp_set, tmp_mask)
    train_set = tmp_set[0:int(_train_lenth * _tv_ratio)]
    dynamics_mask = tmp_mask[0:int(_train_lenth * _tv_ratio)]

    #val_set = tmp_set[int(_train_lenth * _tv_ratio):]
    #val_mask = tmp_mask[int(_train_lenth * _tv_ratio):]

    rand_idx = random.sample(range(0, _train_lenth), val_num)

    val_set = tmp_set[rand_idx]
    val_mask = tmp_mask[rand_idx]


    ab_event_set, ab_dynamic_mask = length_norm(abnormal_set, ab_length_list, max_length)

    # print("[TRAIN]:%d samples (shortest log: %d\tlongest log: %d)"% (len(train_set),min_length,max_length))

    # train_iter, veal_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    # de_size, en_size = len(DE.vocab), len(EN.vocab)
    # print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
    #       % (len(train_iter), len(train_iter.dataset),
    #          len(test_iter), len(test_iter.dataset)))

    validation_loader = ARMIQ_TRAIN_DATALOADER(val_set, val_mask)
    armiq_trainloader = ARMIQ_TRAIN_DATALOADER(train_set, dynamics_mask)

    trainloader = DataLoader(armiq_trainloader, shuffle=True, batch_size=32, num_workers=4)
    valloader = DataLoader(validation_loader, shuffle=False, batch_size=32, num_workers=4)

    #combine part of the normal set and the abnormal set.
    rand_index_for_normal = random.sample(range(3,len(train_set)),int(len(train_set)*0.6))
    eval_set = np.concatenate((ab_event_set,train_set[rand_index_for_normal,:]),axis=0)
    eval_annotation = np.concatenate((np.ones(len(ab_event_set)),np.zeros(len(rand_index_for_normal))),axis=0)
    eval_mask = np.concatenate((ab_dynamic_mask,dynamics_mask[rand_index_for_normal]),axis=0)


    armiq_testloader = ARMIQ_EVAL_DATALOADER(eval_set,eval_annotation,eval_mask)

    evalloader = DataLoader(armiq_testloader,shuffle=True,batch_size=20,num_workers=1)

    print("[!] Instantiating models...")
    encoder1 = Encoder(log_numbers, embed_size, hidden_size, n_layers=2, dropout=0.5,cell_type='GRU')
    decoder1 = Decoder(hidden_size, hidden_size_dec, n_layers=1, dropout=0.5,cell_type='GRU')

    encoder2 = Encoder(log_numbers, embed_size, hidden_size, n_layers=2, dropout=0.5, cell_type='GRU')
    decoder2 = Decoder(hidden_size, hidden_size_dec, n_layers=1, dropout=0.5, cell_type='GRU')

    rvae1 = RVAE(encoder1,decoder1).cuda()

    rvae2 = RVAE(encoder2,decoder2).cuda()


    _model_list1 = ['./save/recon_only/rae_10.pt', './save/recon_only/rae_20.pt',
                   './save/recon_only/rae_30.pt']
    for _index, _path in enumerate(_model_list1):
        print(_path)
        rvae1.load_state_dict(torch.load(_path))
        validation(rvae1, valloader, (_index + 1) * 10,recon=True)


    _model_list2 = ['./save/m_cell/gru/tmp/arae_gru_10.pt','./save/m_cell/gru/tmp/arae_gru_20.pt','./save/m_cell/gru/tmp/arae_gru_30.pt']
    for _index, _path in enumerate(_model_list2):
        print(_path)
        rvae2.load_state_dict(torch.load(_path))
        validation(rvae2,valloader,(_index+1)*10)


if __name__=="__main__":
    try:
        main()
        summary.close()
    except KeyboardInterrupt as e:
	    print("[Stop]",e)
