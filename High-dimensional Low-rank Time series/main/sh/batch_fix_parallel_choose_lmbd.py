import subprocess
import configargparse as argparse
import sys
from itertools import chain
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--C_T0',type=float,default=0.5)
parser.add_argument('--C_s',type=float,default=0.5)
parser.add_argument('--C_l',type=float,default=1)
parser.add_argument('--rho',type=float,default=0.7)
parser.add_argument('--dgp',type=str,choices=['ar','arma','season_ar','season_arma','mix'],default='arma')
parser.add_argument('--dgp_p',type=int,default=1)
parser.add_argument('--dgp_r',type=int,default=4)
parser.add_argument('--dgp_s',type=int,default=1)
parser.add_argument('--task',type=str,choices=['rate','convergence'],required=True)
parser.add_argument('--n_rep',type=int,default=100,help='number of replications')
args = parser.parse_args()

rank = args.dgp_r + 2*args.dgp_s
if args.dgp == 'arma':
    if args.rho == 0.7 and args.dgp_r == 4:
        # T_list = [800]*36
        # T0_list = [25,50,100,200]*9
        # lmda_base = [2.0e-3,2.5e-3,3.0e-3,3.5e-3,4.0e-3,4.5e-3,5.0e-3,5.5e-3,6.0e-3] #len 3
        # lmda_list = [value for value in lmda_base for _ in range(4)] 
        T_list = [800]*4
        T0_list = [100,200]*2
        lmda_base = [0.003,0.004] #len 3
        lmda_list = [value for value in lmda_base for _ in range(2)] 
        # lmda_list = [0.0025,0.004,0.0035]
        # lmda_list = [0.002,0.003,0.004,0.005,0.0025,0.0035,0.0045,0.0055,0.003,0.004,0.005,0.006,0.0035,0.0045,0.0055,0.0065,0.004,0.005,0.006,0.007,0.0045,0.0055,0.0065,0.0075,0.005,0.006,0.007,0.008]
    # elif args.rho == 0.5 and args.dgp_r == 4:
    #     T_list = [300,500,800,1700]
    #     T0_list = [25,33,42,61]
    #     s_list = [8,8,9,10]
    # elif args.rho == 0.3 and args.dgp_r == 4:
    #     T_list = [200,300,600,1000]
    #     T0_list = [21,25,36,47]
    #     s_list = [4,4,5,5]
    elif args.rho == 0.7 and args.dgp_r == 3:
        T_list = [500,1000,1500,2000]*13
        T0_list = [33,43,58,67]*13
        lmbd_list = np.arange()
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-4,6)]))
    elif args.rho == 0.7 and args.dgp_r == 5:
        T_list = [500,1000,1500,2000]*13
        T0_list = [33,43,58,67]*13
        base_s_list = [12,12,12,12]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-4,6)]))
elif args.dgp == 'mix':
    if args.dgp_r == 4:
        T_list = [400,500,900,1900]*5
        T0_list = [30,33,45,65]*5
        base_s_list = [10,10,10,11]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(3,8)]))
    elif args.dgp_r == 3:
        T_list = [300,400,700,1400]*15
        T0_list = [25,30,39,56]*15
        base_s_list = [10,10,10,11]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-6,9)]))
    elif args.dgp_r == 2:
        T_list = [200,300,500,900]*15
        T0_list = [21,25,33,45]*15
        base_s_list = [10,10,10,10]
        s_list = list(chain.from_iterable([[i+j for i in base_s_list] for j in range(-6,9)]))
elif args.dgp == 'season_ar':
    if args.rho == 0.7 and args.dgp_r == 4:
        T_list = [800]*44
        T0_list = [25,50,100,200]*11
        lmda_base = [0.001,0.002,0.005,0.008,0.01,0.012,0.014,0.015,0.016,0.018,0.02] #len 3
        lmda_list = [value for value in lmda_base for _ in range(4)] 
if args.task == 'rate':
    exp_name = f'NewDGP_100_{args.dgp}_optimal_lmbd_change_T0_N20_5'


# for (p,r,dgp_s) in [(1,1,1),(0,1,1),(0,1,0),(0,0,1)]:
#     for T,T0,s in zip(T_list,T0_list,s_list):
#         for init in ['zero','true','spec']:
#             command = ['python3.9','run.py','--dgp_p',str(p),'--dgp_r',str(r),'--dgp_s',str(dgp_s),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--name',exp_name]
#             print(' '.join(command))
#             subprocess.run(command,stdout=None)

# rate
    for T,T0,lmda in zip(T_list,T0_list,lmda_list):
        # for init in ['GD']:
        #     command = ['python3.9','../train.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task',args.task]
        #     print(' '.join(command))
        #     subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        for init in ['true']:
            command = ['python3.9','../train.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--T',str(T),'--lmda',str(lmda),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task',args.task,'--stop_method','loss','--schedule','none','--lr','1e-3','--n_rep',str(args.n_rep),'--N','20','--thresholding_option','soft','--seed','800']
            print(' '.join(command))
            subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

# convergence
# elif args.task == 'convergence':
#     exp_name = 'convergence_test'
#     for T,T0,s,gamma in zip(T_list,T0_list,s_list,gamma_list):
#         for init in ['zero']:
#             command = ['python3.9','run.py','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--dgp_s',str(args.dgp_s),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--s_select','fnorm','--s_thres',str(gamma), '--visualize','--task',args.task]
#             print(' '.join(command))
#             subprocess.run(command,stdout=None)