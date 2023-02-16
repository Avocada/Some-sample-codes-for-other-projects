import subprocess
import configargparse as argparse
import sys


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
args = parser.parse_args()

rank = args.dgp_r + 2*args.dgp_s
if args.dgp == 'arma':
    if args.rho == 0.7 and args.dgp_r == 4:
        T_list = [600,1100,1700,3900]
        T0_list = [36,49,61,93]
        s_list = [17,19,20,23]
    elif args.rho == 0.5 and args.dgp_r == 4:
        T_list = [300,500,800,1700]
        T0_list = [25,33,42,61]
        s_list = [8,8,9,10]
    elif args.rho == 0.3 and args.dgp_r == 4:
        T_list = [200,300,600,1000]
        T0_list = [21,25,36,47]
        s_list = [4,4,5,5]
    elif args.rho == 0.7 and args.dgp_r == 3:
        T_list = [500,800,1300,2800]
        T0_list = [33,42,54,79]
        s_list = [17,18,19,22]
    elif args.rho == 0.7 and args.dgp_r == 2:
        T_list = [300,500,800,1800]
        T0_list = [25,33,42,63]
        s_list = [15,17,18,20]
elif args.dgp == 'season_ar':
    args.dgp_p = 4
    if args.dgp_r == 4:
        T_list = [209,309,409,809]
        T0_list = [9,9,9,9]
        s_list = [7,7,7,7]
    elif args.dgp_r == 3:
        T_list = [159,209,309,609]
        T0_list = [9,9,9,9]
        s_list = [7,7,7,7]
    elif args.dgp_r == 2:
        T_list = [109,159,209,409]
        T0_list = [9,9,9,9]
        s_list = [7,7,7,7]
elif args.dgp == 'mix':
    if args.dgp_r == 4:
        T_list = [400,500,900,1900]
        T0_list = [30,33,45,65]
        s_list = [10,10,10,11]
    elif args.dgp_r == 3:
        T_list = [300,400,700,1400]
        T0_list = [25,30,39,56]
        s_list = [10,10,10,11]
    elif args.dgp_r == 2:
        T_list = [200,300,500,900]
        T0_list = [21,25,33,45]
        s_list = [8,8,8,9]


if args.task == 'rate':
    exp_name = f'{args.dgp}_diffhalf_true'


# for (p,r,dgp_s) in [(1,1,1),(0,1,1),(0,1,0),(0,0,1)]:
#     for T,T0,s in zip(T_list,T0_list,s_list):
#         for init in ['zero','true','spec']:
#             command = ['python3.9','run.py','--dgp_p',str(p),'--dgp_r',str(r),'--dgp_s',str(dgp_s),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--name',exp_name]
#             print(' '.join(command))
#             subprocess.run(command,stdout=None)

# rate
    for T,T0,s in zip(T_list,T0_list,s_list):
        # for init in ['GD']:
        #     command = ['python3.9','../train.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task',args.task]
        #     print(' '.join(command))
        #     subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        if float(sys.version[:3])<3.9:
            cmd = 'python'
        else:
            cmd = 'python3.9'
        for init in ['noisetrue']:
            command = [cmd,'../train.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task',args.task,'--stop_method','Adiff','--schedule','half','--lr','1e-3']
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