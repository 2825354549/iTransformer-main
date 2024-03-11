import argparse
import os
import time

import pandas as pd
import torch
# from utils.tools import setup_seed
# from exp.exp_main import Exp_LSTM
import matplotlib.pyplot as plt
# from run_lstm import initial
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error



if __name__ == '__main__':
    # setup_seed(2023)
    start_time = time.time()

    # args = initial()

    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    preds = np.load('./results/test_iInformer_custom_MS_ft48_sl0_ll1_pl1024_dm8_nh2_el1_dl1024_df1_fctimeF_ebTrue_dttest_projection_0'+'/pred.npy')
    trues = np.load('./results/test_iInformer_custom_MS_ft48_sl0_ll1_pl1024_dm8_nh2_el1_dl1024_df1_fctimeF_ebTrue_dttest_projection_0'+ '/true.npy')
    # print(preds[:,0, :].shape, trues[:,0, :].shape)  (7456, 1) (7456, 1)
    # plt.plot(preds[:,0, :], label='prediction', color='red')
    # plt.plot(trues[:,0, :], label='real', color='blue')
    # print(np.mean(mean_absolute_error(preds[:,0, :],trues[:,0, :])))
    # plt.legend()
    # plt.show()
    #
    # print('prediction')

    maes = []
    mses = []
    rmses = []
    combined_pred = np.zeros((1, 1))
    combined_true = np.zeros((1, 1))
    for i in range(0,preds.shape[0],1):
    # for i in range(0, 30, args.pred_len):

        pred =preds[i,:,:]
        true =trues[i,:,:]
        # print(pred.shape)
        combined_pred = np.vstack((combined_pred, pred))
        combined_true = np.vstack((combined_true, true))

    # true = np.array(trues[i,:,:].view(-1,1).numpy())
        mae = mean_absolute_error(true,pred)
        mse = mean_squared_error(true,pred)
        rmse = np.sqrt(mse)

        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)

        # print(mae)
        # print(mse)
        # print(rmse)
        # plt.figure()
        # plt.plot(preds[i,:,:], label='prediction',color = 'red')
        # plt.plot(trues[i,:,:], label='real',color = 'blue')
        # plt.legend()
        # # plt.ylim(-1, 1)
        # plt.show()
        # print(i)
        # print('prediction')
    maes = np.mean(maes)
    mses = np.mean(mses)
    rmse = np.mean(rmses)
    plt.plot(combined_pred[1:, :], label='prediction', color='red')
    plt.plot(combined_true[1:, :], label='real', color='blue')
    plt.legend()
    plt.show()

    print('mse',mses)
    print('mae',maes)
    print('rmse',rmse)

    # 0.04390015
    # 0.0045152954
    # 0.059646275

    # 0.020686667
    # 0.00091545854
    # 0.027666045

    # 0.003562987
    # 0.044603445
    # 0.05357843


    # mse: 0.0036741739604622126
    # mae: 0.045989472419023514
    # rse: 0.06058695167303085
    #only gru
    # 0.0040352684
    # 0.04479447
    # 0.05146695

    #  mse:0.004049456212669611
    #  mae:0.04452585428953171
    #  rse:0.06360592693090439
    # lstm
    # mse:0.0028314802329987288
    #  mae:0.03736837953329086
    #  rse:0.053187064826488495

    # mse:0.0009745928109623492
    # mae:0.01909671537578106
    # rse:0.031204041093587875

























































