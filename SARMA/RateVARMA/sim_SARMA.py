import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import tensorly as tl
from help_func import *
from DGP import *
from ALS import *

import sys, getopt

help_man = """
sys arg:
-e (str) learning rate
-n (int) number of repeated experiments
-m (double) random seed
-l (double) threshold value of stopping criteria, apply to both methods

-N (int)
-r (int)
-s (int)
-p (int)
"""

def main(argv):
    # model setting
    N = 10
    p = 0
    r = 1
    s = 1
    burn = 200
    r1 = r + 2*s
    r2 = r+2*s
    n_iter = 100
    seed = 0

    # hyper parameters default values
    P_est = 5
    flag_true_G = True
    stop_method = 'SepEst'
    lr_tensor = 5e-3
    lr_omega = 1e-1
    beta = 0
    max_iter = 1000
    stop_thres = 1e-2

    try:
        opts, args = getopt.getopt(argv,"he:n:m:l:N:r:s:p:")
    except getopt.GetoptError:
        print('No such options')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_man)
            sys.exit()
        elif opt in ("-e"):
            lr_omega = float(arg)
        elif opt in ("-n"):
            n_iter = int(arg)
        elif opt in ("-m"):
            seed = int(arg)
        elif opt in ("-l"):
            stop_thres = float(arg)
        elif opt in ("-N"):
            N = int(arg)
        elif opt in ("-r"):
            r = int(arg)
        elif opt in ("-s"):
            s = int(arg)
        elif opt in ("-p"):
            p = int(arg)

    # use model setting to automatically select sample size
    sample_size = get_sample_size_VARMADGP(N,p,r,s)
    # sample_size = [1000]
    T = sample_size[-1]
    P_dgp = T+burn
    r1 = r + 2*s
    r2 = r+2*s
    # r1 = r2 = 4

    est_res = np.zeros((6*len(sample_size)+1,n_iter))
    # np.savetxt('result/'+str(T)+'_trueG_stopLoss_5e-4,1e-4,0.9.csv', est_res.T, fmt='%10.7f',delimiter=',')
    for iter in range(n_iter):
        np.random.seed(iter+seed)
        print('Iter: ', iter)
        # generate max length of y
        y,lmbd,gamma,theta,G,L = DGP_VARMA(N,T,burn,p,r,s)
        # print('True loss: ',loss(y,G,L,T))
        true_A = tl.tenalg.mode_dot(G,L,2)
        est_res[-1,iter] = np.linalg.norm(tensor_op.unfold(true_A[:,:,:T],1),ord='fro')
        # print(lmbd)
        # print("True A's norm: ", np.linalg.norm(tensor_op.unfold(true_A,1),ord='fro'))
        for l in range(len(sample_size)):
            flag_boundary = 0
            ss = sample_size[l]
            A,lmbd_est,gamma_est,theta_est,G_est,Loss,flag_maxiter = ALS(y[:,(T-ss):],p,r,s,r1,r2,N,ss,P_est,200,lr_omega,np.array(lmbd),np.array(gamma),np.array(theta),true_A[:,:,:T],G,stop_thres,flag_true_G,stop_method)
            if 0.9 in np.concatenate([lmbd_est,gamma_est]) or -0.9 in lmbd_est or 0.1 in gamma_est:
                flag_boundary = 1
            est_res[6*l,iter] = np.sqrt(np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:ss],1),ord='fro')**2 + np.linalg.norm(tensor_op.unfold(true_A[:,:,ss:],1),ord='fro')**2)
            est_res[6*l+1,iter] = np.sqrt(np.linalg.norm(np.array(lmbd)-np.array(lmbd_est),ord=2)**2 + np.linalg.norm(np.array(gamma)-np.array(gamma_est),ord=2)**2 + np.linalg.norm(np.array(theta)-np.array(theta_est),ord=2)**2)
            est_res[6*l+2,iter] = np.linalg.norm(tensor_op.unfold(G-G_est,1),ord='fro')
            # print(np.linalg.norm(tensor_op.unfold(G-G_est,1),ord='fro'))
            est_res[6*l+3,iter] = Loss
            est_res[6*l+4,iter] = flag_boundary
            est_res[6*l+5,iter] = flag_maxiter

            

            # disturbation on true value
            # d = r+2*s
            # disturb_lmbd = lmbd + disturb * (np.random.rand(r)-0.5) * 1e-3
            # disturb_L = get_L(disturb_lmbd,gamma,theta,N,r,s,T,p)
            # disturb_G = G + disturb * (np.random.rand(N,N,d)-0.5) * 1e-3
            # disturb_A = mode_dot(disturb_G,disturb_L,2)
            # check distance
            # dist = np.linalg.norm(tensor_op.unfold(disturb_A - true_A[:,:,:T],1), ord='fro')
            # if dist > 1e-2:
            #     pass
            # use truncated sample to fit

            # A = GD(y[:,(T-ss):],p,r,s,r1,r2,N,ss,P_est,lr_tensor,lr_omega,beta,max_iter,lmbd,gamma,theta,true_A[:,:,:T],G,flag_true_G,disturb,stop_method)
            # est_res[l,iter] = np.sqrt(np.linalg.norm(tensor_op.unfold(A-true_A[:,:,:ss],1),ord='fro')**2 + np.linalg.norm(tensor_op.unfold(true_A[:,:,ss:],1),ord='fro')**2)
    #         with open('result/'+str(ss)+'_A.npy', 'wb') as f:
    #             np.save(f,A)
    # with open('result/trueA.npy', 'wb') as f:
    #     np.save(f,true_A)
        

    f=open('result/N'+str(N)+'_p'+str(p)+'_r'+str(r)+'_s'+str(s)+'S'+str(stop_thres)+'seed'+str(seed)+'.csv','a')
    np.savetxt(f, sample_size, fmt='%10.7f',delimiter=',',newline=',')
    f.write("\n")
    np.savetxt(f, est_res.T, fmt='%10.7f',delimiter=',')
    f.close()

"""
est_res fmt:

"""

if __name__ == "__main__":
   main(sys.argv[1:])
