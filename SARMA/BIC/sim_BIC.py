import os
os.environ["OMP_NUM_THREADS"] = "1"
from ADMM import *
import numpy as np
import tensorly as tl
from help_func import *
from DGP import *
from ALS import ALS
import timeit

import sys, getopt

help_man = """
sys arg:
-n (int) number of repeated experiments
-l (double) threshold value of stopping criteria, apply to both methods
-r (double) hyper-parameter for nuclear-norm penalization
-m (int) random seed
-a (double) signal strength of sdgp

-N (int)
-r (int)
"""

def main(argv):

    # model setting
    N = 10
    p = 0
    r = 1
    s = 1
    T = 1000
    burn = 200
    r1 = r+2*s
    r2 = r+2*s
    n_iter = 1
    P_dgp = T+burn

    # candidate BIC parameters
    p_set = [0,1]
    r_set = [0,1,2]
    s_set = [0,1,2]
    

    # hyper parameters default values
    P_est = 5
    flag_true_G = True
    stop_method = 'SepEst'
    lr_tensor = 5e-3
    lr_omega = 1
    beta = 0
    max_iter = 6
    stop_thres = 5e-3
    # c = np.sqrt(N*P_lag*np.log(T-P_lag)/10/(T-P_lag))
    m= 4
    seed = 0
    signal =0.8


    try:
        opts, args = getopt.getopt(argv,"hp:s:n:T:l:N:r:m:a:")
    except getopt.GetoptError:
        print('No such options')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_man)
            sys.exit()
        elif opt in ("-p"):
            p = int(arg)
        elif opt in ("-s"):
            s = int(arg)
        elif opt in ("-n"):
            n_iter = int(arg)
        elif opt in ("-T"):
            T = int(arg)
        elif opt in ("-l"):
            stop_thres = float(arg)
        elif opt in ("-N"):
            N = int(arg)
        elif opt in ("-r"):
            r = int(arg)
        elif opt in ("-m"):
            seed = int(arg)
        elif opt in ("-a"):
            signal = float(arg)

    # use model setting to automatically select sample size
    r1 = r+2*s
    r2 = r+2*s
    if p == 1 and  r == 1 and s == 0:
        r1 = r2 = 2
    # P_lag = int(np.sqrt(T))
    P_lag = int(T**(1/3))
    # P_lag = 10
    P_dgp = T+burn
    # for 010
    # if T == 1000:
    #     c1 = 0.05
    #     c2 = 0.1
    #     rho = 1e-2
    # if T == 1500:
    #     c1 = 0.1
    #     c2 = 0.1
    #     rho = 2e-2
    c = np.sqrt(N*P_lag*np.log(T-P_lag)/10/(T-P_lag))
    c1 = c
    c2 = c
    # rho_set = np.linspace(0.1,0.5,num=10)
    rho_set = np.logspace(-2,0,num=10)
    # rho_set = [0.03,0.02,0.01,0.005]

    Loss_list = np.zeros((len(p_set)*len(r_set)*len(s_set)-1,n_iter))
    est_Loss_list = np.zeros((len(p_set)*len(r_set)*len(s_set)-1,n_iter))
    BIC_list = np.zeros((len(p_set)*len(r_set)*len(s_set)-1,n_iter))
    count = -1
    rank = np.zeros((n_iter,2))
    sing_values = np.zeros((n_iter,2*N))
    summary_table = np.zeros(10)
    start_time = timeit.default_timer()
    error_seed = np.zeros((len(p_set)*len(r_set)*len(s_set)-1,n_iter))
    for iter in range(n_iter):
        print(iter)
        np.random.seed(iter+seed)
        # y,lmbd,gamma,theta,G,L = DGP_BIC(N,T,burn,p,r,s,r1,r2,P_dgp,st_control=10)
        y,lmbd,gamma,theta,G,L = DGP_AUTO(N,T,burn,p,r,s,signal)
        true_A = tl.tenalg.mode_dot(G,L,2)
        
        # create X (regressors)
        X = np.zeros((N*P_lag,T-P_lag))
        for i in range(P_lag):
            X[i*N:i*N+N,:] = y[:,P_lag-i-1:T-i-1]
        # create Y (response)
        Y = y[:,P_lag:]
        # initialization
        A0 = (X@Y.T).T @ np.linalg.inv(X @ X.T)
        # fold A into a tensor
        # A0 = np.array(tensor_op.fold(A0,(N,N,P_lag),1))

        min_err = np.inf
        pre_A = A0
        for rho in rho_set:
            forecast_err = 0
            for t in range(int(T/10)-P_lag):
                A_NN = ADMM_2(Y[:,:int(t+9*T/10)],X[:,:int(t+9*T/10)],pre_A,1,rho,N,P_lag,T,100,true_A)
                forecast_err += np.linalg.norm(Y[:,int(t+9*T/10)] - A_NN @ X[:,int(t+9*T/10)],)**2
                pre_A = A_NN
            if forecast_err < min_err:
                min_rho = rho
                min_err = forecast_err
                best_A0 = np.copy(A_NN)
            Loss = np.sum(np.linalg.norm(Y - A_NN @X,ord=2,axis=0)**2) / T
            # print("lmbd:",rho,"forecast err: ",forecast_err,"loss: ",Loss)
            
        # print(min_err)
        A_NN = ADMM_2(Y,X,best_A0,1,min_rho,N,P_lag,T,100,true_A)  
        _,s1,_ = np.linalg.svd(A_NN,full_matrices=False)
        A_NN = tensor_op.unfold(tensor_op.fold(A_NN,(N,N,P_lag),1),2)
        _,s2,_ = np.linalg.svd(A_NN,full_matrices=False)
        
        # c1 = m*np.average(s1)
        # c2 = m*np.average(s2)
        ratio1 = np.zeros(9)
        ratio2 = np.zeros(9)
        for i in range(1,10):
            ratio1[i-1] = (s1[i]+c1)/(s1[i-1]+c1)
            ratio2[i-1] = (s2[i]+c2)/(s2[i-1]+c2)
        r1_est = np.argmin(ratio1)+1
        r2_est = np.argmin(ratio2)+1

        if r1_est == r1 and r2_est == r2:
            summary_table[0] += 1



        # calulate BIC
        count = -1
        min_BIC_value = min_BIC_value_est_rank = np.inf
        for p_ in p_set:
            for r_ in r_set:
                for s_ in s_set:
                    if p_+r_+s_ == 0:
                        continue
                    count = count +1 
                    # BIC_list[0,count] = p_
                    # BIC_list[1,count] = r_
                    # BIC_list[2,count] = s_

                    # Initial values of omega
                    if r_ == 2:
                        lmbd = [0.3,-signal]
                    elif r_ == 1:
                        lmbd = [-signal]
                    elif r_ == 0:
                        lmbd = []
                    
                    if s_ == 0:
                        gamma = []
                        theta = []
                    elif s_ == 1:
                        gamma = [signal]
                        theta = [np.pi/4]
                    elif s_ == 2:
                        gamma = [0.3,signal]
                        theta = [-np.pi/4,np.pi/4]
                    # use true rank
                    NofP = N*(r1+r2) + r1*r2*(p_+r_+2*s_)
                    # print(p_,r_,s_)
                    # L_init = get_L(lmbd,gamma,theta,N,r_,s_,P_lag,p_)
                    # A_NN = tensor_op.fold(A_NN,(N,N,P_lag),1).numpy()
                    # G_init = mode_dot(A_NN,np.linalg.inv(L_init.T@ L_init)@ L_init.T,2)
                    # print(G_init.shape)
                    if p_ == p and r_ == r and s_ == s:
                        A_true_rank,lmbd_est,gamma_est,theta_est,G_est,Loss_true_rank,_,flag_ValueError = ALS(y,p_,r_,s_,r1,r2,N,T,P_est,100,lr_omega,lmbd,gamma,theta,true_A,G,stop_thres=stop_thres,flag_true_G=True,stop_method=stop_method)
                    else:
                        A_true_rank,lmbd_est,gamma_est,theta_est,G_est,Loss_true_rank,_,flag_ValueError = ALS(y,p_,r_,s_,r1,r2,N,T,P_est,100,lr_omega,lmbd,gamma,theta,true_A,G,stop_thres=stop_thres,flag_true_G=False,stop_method=stop_method)
                    print(np.log(Loss_true_rank))
                    Loss_list[count,iter] = np.log(Loss_true_rank)
                    BIC_value = np.log(Loss_true_rank) + 0.1 * NofP *np.log(T)/T
                    if BIC_value < min_BIC_value:
                        min_BIC_value = BIC_value
                        p_BIC = p_
                        r_BIC = r_
                        s_BIC = s_

                    # use estimated rank
                    if r1_est == r1 and r2_est == r2:
                        continue
                    NofP = N*(r1_est+r2_est) + r1_est*r2_est*(p_+r_+2*s_)
                    A_est_rank,lmbd_est,gamma_est,theta_est,G_est,Loss_est_rank,_,flag_ValueError = ALS(y,p_,r_,s_,r1_est,r2_est,N,T,P_est,100,lr_omega,lmbd,gamma,theta,true_A,G,stop_thres=stop_thres,flag_true_G=False,stop_method=stop_method)
                    est_Loss_list[count,iter] = np.log(Loss_est_rank)
                    if flag_ValueError == 1:
                        error_seed[count,iter] = 1
                    BIC_value = np.log(Loss_est_rank) + 0.1 * NofP *np.log(T)/T
                    if BIC_value < min_BIC_value_est_rank:
                        min_BIC_value_est_rank = BIC_value
                        p_BIC_est_rank = p_
                        r_BIC_est_rank = r_
                        s_BIC_est_rank = s_

        # BIC assuming ranks known
        if p_BIC == p and r_BIC == r and s_BIC == s:
            summary_table[1] += 1
        elif p_BIC == p and r_BIC >=r and s_BIC >= s:
            summary_table[3] += 1
        else:
            summary_table[2] += 1

        # BIC if ranks are correct
        # if T != 200 and T!= 300:
        #     continue

        if r1_est == r1 and r2_est == r2:
            if p_BIC == p and r_BIC == r and s_BIC == s:
                summary_table[4] += 1
            elif p_BIC == p and r_BIC >=r and s_BIC >= s:
                summary_table[6] += 1
            else:
                summary_table[5] += 1

        # BIC if ranks are wrong
        else:
            if p_BIC_est_rank == p and r_BIC_est_rank == r and s_BIC_est_rank == s:
                summary_table[7] += 1
            elif p_BIC_est_rank == p and r_BIC_est_rank >=r and s_BIC_est_rank >= s:
                summary_table[9] += 1
            else:
                summary_table[8] += 1
                    


        # f=open('result/H_Loss_N'+str(N)+'_p'+str(p)+'_r'+str(r)+'_s'+str(s)+'.csv','a')
        # np.savetxt(f, Loss_list.T[iter,:], fmt='%10.7f',delimiter=',')
        # f.close()
        # f=open('result/H_BIC_N'+str(N)+'_p'+str(p)+'_r'+str(r)+'_s'+str(s)+'.csv','a')
        # np.savetxt(f, BIC_list.T[iter,:], fmt='%10.7f',delimiter=',')
        # f.close()

    # stop_time = timeit.default_timer()
    f=open('result/summary/Summary_seed'+str(seed)+'_T'+str(T)+'_sig'+str(signal)+'_set'+str(p)+str(r)+str(s)+'.csv','w')
    np.savetxt(f, summary_table, fmt='%10.7f',delimiter=',')
    f.close()

    f=open('result/loss/Loss_seed'+str(seed)+'_T'+str(T)+'_sig'+str(signal)+'_set'+str(p)+str(r)+str(s)+'.csv','w')
    np.savetxt(f, Loss_list.T, fmt='%10.7f',delimiter=',')
    # f.write(str(stop_time-start_time))
    f.write("\n")
    np.savetxt(f, est_Loss_list.T, fmt='%10.7f',delimiter=',')
    f.close()

    f=open('result/errorlist/Error_seed'+str(seed)+'_T'+str(T)+'_sig'+str(signal)+'_set'+str(p)+str(r)+str(s)+'.csv','w')
    np.savetxt(f,error_seed.T, fmt='%10.7f',delimiter=',',newline='\n')
    f.close()

    f=open('result/rank/Rank_seed'+str(seed)+'_T'+str(T)+'_sig'+str(signal)+'_set'+str(p)+str(r)+str(s)+'.csv','w')
    np.savetxt(f, rank, fmt='%10.7f',delimiter=',',newline='\n')
    f.write("\n")
    f.close()

                    
if __name__ == "__main__":
    main(sys.argv[1:])
