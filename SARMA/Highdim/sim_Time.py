import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import tensorly as tl
from help_func import *
from DGP import *
from ALS import *
import timeit
import statsmodels.api as sm

import sys, getopt

help_man = """
sys arg:
-n (int) number of repeated experiments
-m (double) random seed
-l (double) threshold value of stopping criteria, apply to both methods

-N (int)
-r (int)
-s (int)
-p (int)
-T (int)
"""

def main(argv):
    # model setting
    N = 40
    p = 0
    r = 0
    s = 0
    burn = 200
    n_iter = 10
    seed = 0

    # hyper parameters default values
    P_est = 5
    flag_true_G = True
    stop_method = 'SepEst'
    lr_tensor = 5e-3
    lr_omega = 3e-1
    # lr_omega = 3e-1 # used lr
    beta = 0
    max_iter = 5
    stop_thres = 1e-2
    T = 1000

    try:
        opts, args = getopt.getopt(argv,"he:n:m:l:N:r:s:p:T:")
    except getopt.GetoptError:
        print('No such options')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_man)
            sys.exit()
        elif opt in ("-e"):
            stop_method = arg
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
        elif opt in ("-T"):
            T = int(arg)

    # use model setting to automatically select sample size
    r1 = r2 = r+2*s

    est_res = np.zeros((6+1,n_iter))
    err_sarma = np.zeros((n_iter,2))
    time = np.zeros((n_iter,2))

    # np.savetxt('result/'+str(T)+'_trueG_stopLoss_5e-4,1e-4,0.9.csv', est_res.T, fmt='%10.7f',delimiter=',')
    for iter in range(n_iter):
        np.random.seed(iter+seed)
        print('Iter: ', iter)
        # generate max length of y
        y,lmbd,gamma,theta,G,L,Phi,Theta = DGP_TIME(N,T+1,burn,0,r,s)
        y_target = y[:,T]
        y = y[:,:T]
        x = np.reshape(np.flip(y,axis=1),(-1,1),order='F')
        # print('True loss: ',loss(y,G,L,T))
        true_A = tl.tenalg.mode_dot(G,L,2)
        est_res[-1,iter] = np.linalg.norm(tensor_op.unfold(true_A[:,:,:T],1),ord='fro')
        # print(lmbd)
        # print("True A's norm: ", np.linalg.norm(tensor_op.unfold(true_A,1),ord='fro'))
        
        flag_boundary = 0

        # start_sarma = timeit.default_timer()
        # A_est,lmbd_est,gamma_est,theta_est,G_est,Loss,flag_maxiter = ALS(y,p,r,s,r1,r2,N,T,P_est,max_iter,lr_omega,np.array(lmbd),np.array(gamma),np.array(theta),true_A[:,:,:T],G,stop_thres)
        # # print(A_est)
        # end_sarma = timeit.default_timer()

        # est_res[0,iter] = flag_maxiter
        # y_forecast_sarma = tensor_op.unfold(A_est,1).numpy() @ x
        # # print(y_forecast_sarma)
        # # print(y_target)
        # err_sarma[iter,0] = np.linalg.norm(y_forecast_sarma-y_target[:,np.newaxis])
        # err_sarma[iter,1] = np.linalg.norm(y_forecast_sarma-y_target[:,np.newaxis],ord=1)
        # time[iter,0] = end_sarma - start_sarma

        params = np.array(  [0]*N+Phi.ravel().tolist()  + Theta.ravel().tolist() + [1]*N).ravel()
        # start_varma = timeit.default_timer()
        VARMA = sm.tsa.VARMAX(y.T, order=(0,1),error_cov_type="diagonal", return_params=True,includes_fixed=True)
        # print(VARMA.start_params.shape)
        # res_varma = VARMA.fit(disp=False)
        start_varma = timeit.default_timer()
        res_varma = VARMA.fit(start_params = VARMA.start_params,disp=False,maxiter=5)
        # print(res_varma.mle_settings)
        end_varma = timeit.default_timer()
        # y_forecast = res_varma.forecast(1)
        # err_varma[iter,0] = np.linalg.norm(y_forecast.T-y_target[:,np.newaxis])
        # err_varma[iter,1] = np.linalg.norm(y_forecast.T-y_target[:,np.newaxis],ord=1)
        time[iter,1] = end_varma - start_varma

        print(time[iter,:])

    # print(np.mean(err_sarma,0))
    print(np.mean(time,0))
        

    # f=open('result/avg/4'+'T'+str(T)+"_r"+str(r)+'_s'+str(s)+'.csv','w')
    # f.write(str(np.mean(err_sarma,0)))
    # f.write("\n")
    # f.write(str(np.mean(time,0)))
    # f.close()

    # f=open('result/record/4'+'T'+str(T)+"_r"+str(r)+'_s'+str(s)+'.csv','w')
    # np.savetxt(f, np.concatenate([err_sarma,time],axis=1), fmt='%10.7f',delimiter=',')
    # f.close()


"""
est_res fmt:

"""

if __name__ == "__main__":
   main(sys.argv[1:])
