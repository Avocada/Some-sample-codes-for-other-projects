import os
os.environ["OMP_NUM_THREADS"] = "4"
import logging

import numpy as np
import argparse
import tensorly as tl
import pandas as pd
import logging

from dgp import *
from alg_real import train_epoch



######################### File path ##########################
money_loc = "data/money.csv"
RV_loc = "data/RVdata46.csv"
##############################################################

def main():
    # model setting
    step_size = 1e-3

    # hyper parameters default values
    stop_thres = 1e-4

    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_rep',type=int,default=10,help='number of replications')
    parser.add_argument('--a',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--b',type=float,default=1,help='hyperparameter in gradient descent')
    parser.add_argument('--T0', type=int,default=10, help='T0')
    parser.add_argument('--s',type=int,default=4,help='sparsity level')
    parser.add_argument('--lmda',type=float,default=1e-1,help='hyperparameter in gradient descent')
    parser.add_argument('--data',type=str,default='money',choices=['money','RV'],help='name of the dataset')
    parser.add_argument('--r1',type=int,default=3,help='rank at mode 1')
    parser.add_argument('--r2',type=int,default=3,help='rank at mode 2')
    parser.add_argument('--thresholding_option',type=str,default='hard',choices=['hard','soft','none'],
                        help='choose hard thresholding or soft thresholding or no thresholding')
    
    parser.add_argument('--A_init',type=str,choices=['spec','zero'],default='zero')
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--max_iter',type=int,default=5000)
    args = parser.parse_args()

    np.random.seed(0)

    # logging
    wkdir = f'result/{args.data}'
    if not os.path.exists(wkdir):
        os.makedirs('{}/log'.format(wkdir))
        os.makedirs('{}/csv'.format(wkdir))
    logging.basicConfig(filename='{}/log/{}tune.log'.format(wkdir,args.T0), encoding='utf-8', level=logging.CRITICAL,
                            format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    logging.critical(f"Experiment setting:\n --a {args.a} --b {args.b} --T0 {args.T0} --s {args.s} --r1 {args.r1} --r2 {args.r2} --lr {args.lr} --A_init {args.A_init} --max_iter {args.max_iter}")


    ############### read and process data #############
    # Macro40
    if args.data == "money":
        df = pd.read_csv(money_loc)
        y = df.to_numpy()
        y = y[:,1:].astype('float64')
        y = y.T[:,:243]
        N,T = y.shape
        print(T,N)
        y_mean = np.mean(y,1)
        y_sd = np.std(y,1)
        y_normed = (y - y_mean[:,np.newaxis])/y_sd[:,np.newaxis]
        start_ind = 215
    # Realized volatility
    elif args.data == "RV":
        df = pd.read_csv(RV_loc)
        RVraw = df.to_numpy().astype('float64')
        N,T = RVraw.shape
        print(N,T)
        print(RVraw)
        y_mean = np.mean(RVraw, 1)
        y_sd = np.std(RVraw, 1)
        y_normed = (RVraw - y_mean[:, np.newaxis])/ y_sd[:,np.newaxis]
        ttratio = 0.9
        trainsz = int(ttratio * y_normed.shape[1])
        testsz = y_normed.shape[1] - int(ttratio * y_normed.shape[1])
        # print("We use " + str(trainsz) + " for training, and " + str(testsz) + " for testing.")
        start_ind = trainsz

    # rolling forecast
    forecast_err = np.zeros((T-start_ind,2))
    # true_A = pre_G = None
    tar_norm = 0
    Loss_list = np.zeros((T-start_ind))

    for t in range(start_ind,T):
        print("Forecast target: ", t)
        y = y_normed[:,:(t+1)].T
        X = np.zeros((t+1-args.T0,N,args.T0))
        for i in range(args.T0):
            X[:,:,i] = y[(args.T0-1-i):t-i,:]
        y = y[args.T0:,:]
        y_target = y[-1,:]
        y = y[:-1,:]
        x = X[-1,:,:]
        X = X[:-1,:,:]
        
        if args.A_init == 'zero':
            A_init = np.zeros((N,N,args.T0))
        else:
            A_init = None
        A_est,U1,U2,norm_0_idx,_,_ = train_epoch(y=y,X=X,P=args.T0,r1=args.r1,r2=args.r2,a=args.a,b=args.b,s=args.s,lmda=args.lmda,thresholding_option=args.thresholding_option,max_iter=args.max_iter,step_size=args.lr,A_init=A_init,min_loss=stop_thres)
        nonzero_idx = list(np.arange(0,args.T0))
        for i in norm_0_idx:
            nonzero_idx.remove(i)
        y_forecast = np.einsum('NP,iNP->i',x,A_est)
        forecast_err[t-start_ind,0] = np.linalg.norm(y_forecast-y_target)
        forecast_err[t-start_ind,1] = np.linalg.norm(y_forecast-y_target,ord=1)
        tar_norm += np.linalg.norm(y_target[:,np.newaxis])
        print("err: ",forecast_err[t-start_ind,:])

        f=open(f'result/{args.data}/csv/T0{args.T0}rank_'+str(args.r1)+str(args.r2)+"s"+str(args.s)+"a"+str(args.a)+"b"+str(args.b)+'U1.csv','a')
        np.savetxt(f, U1, fmt='%10.7f',delimiter=',')
        f.write("\n")
        f.close()
        f=open(f'result/{args.data}/csv/T0{args.T0}rank_'+str(args.r1)+str(args.r2)+"s"+str(args.s)+"a"+str(args.a)+"b"+str(args.b)+'U2.csv','a')
        np.savetxt(f, U2, fmt='%10.7f',delimiter=',')
        f.write("\n")
        f.close()
        f=open(f'result/{args.data}/csv/T0{args.T0}rank_'+str(args.r1)+str(args.r2)+"s"+str(args.s)+"a"+str(args.a)+"b"+str(args.b)+'choose_lag.csv','a')
        np.savetxt(f, nonzero_idx, fmt='%10.7f',delimiter=',')
        f.write("\n")
        f.close()
        
    logging.critical(f"Mean forecast error: {np.mean(forecast_err,0)}",)
    # stop_time = timeit.default_timer()


    # f=open(f'result/{args.data}/GLasso_FE_rank_'+str(args.r1)+str(args.r2)+"sparse"+str(args.s)+"a"+str(args.a)+"b"+str(args.b)+'.csv','a')
    # f.write(str(np.mean(forecast_err,0)))
    # f.write("\n")
    # np.savetxt(f, forecast_err, fmt='%10.7f',delimiter=',')
    # f.close()

    # f=open(f'result/{args.data}/GLasso_Loss_rank_'+str(args.r1)+str(args.r2)+"sparse"+str(args.s)+"a"+str(args.a)+"b"+str(args.b)+'.csv','a')
    # np.savetxt(f, Loss_list, fmt='%10.7f',delimiter=',')
    # f.close()


if __name__ == "__main__":
   main()
