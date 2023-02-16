import subprocess
import configargparse as argparse


parser = argparse.ArgumentParser()
parser.add_argument('--rho',type=float,default=0.7)
parser.add_argument('--dgp',type=str,choices=['ar','arma','season_ar','season_arma','mix'],default='arma')
parser.add_argument('--dgp_p',type=int,default=1)
parser.add_argument('--dgp_r',type=int,default=4)
parser.add_argument('--dgp_s',type=int,default=1)
parser.add_argument('--N',type=int,default=10)
parser.add_argument('--task',type=str,choices=['rate','convergence'],required=True)
parser.add_argument('--changevar',type=str,choices=['T','rank','s','N'],required=True)
args = parser.parse_args()

rank = args.dgp_r + 2*args.dgp_s
if args.dgp == 'arma':
    if args.rho == 0.7 and args.dgp_r == 4 and args.changevar == 'T':
        T_list = [860,1070,1420,2120,4200]
        T0_list = [43,49,56,69,97]
        # T0_list = [16,17,18,20,24]
        # T0_list = [28,30,33,38,48]
        s_list = [10,10,10,10]
        r_list = [4,4,4,4]
        N_list = [20]*4
    elif args.rho == 0.7 and args.dgp_r == 4 and args.changevar == 'rank':
        T_list = [1200]*4
        T0_list = [51]*4
        s_list = [10]*4
        r_list = [2,3,4,5]
        N_list = [20]*4
    elif args.rho == 0.7 and args.dgp_r == 4 and args.changevar == 'N':
        T_list = [2000]*4
        T0_list = [20]*4
        s_list = [10]*4
        r_list = [4]*4
        N_list = [10,20,30,40]
    elif args.rho == 0.7 and args.dgp_r == 4 and args.changevar == 's':
        s_list = [6,7,8,9,10,11,12,13,14,16,18,20,22]
        T_list = [2000]*len(s_list)
        T0_list = [50]*len(s_list)
        r_list = [4]*len(s_list)
        N_list = [20]*len(s_list)
elif args.dgp == 'season_ar':
    if args.rho == 0.7 and args.dgp_r == 4 and args.changevar == 'T':
        T_list = [410,510,680,1020,2030]
        # T0_list = [9]*5
        # T0_list = [13,14,15,16,20]
        T0_list = [22,23,26,30,37]
        # T0_list = [30,33,39,47,67]
        s_list = [5]*5
        r_list = [4]*5
        N_list = [20]*5
    elif args.rho == 0.7 and args.dgp_r == 4 and args.changevar == 'N':
        T_list = [1200]*4
        T0_list = [31]*4
        s_list = [5]*4
        r_list = [4]*4
        N_list = [10,20,30,40]
    elif args.rho == 0.7 and args.dgp_r == 4 and args.changevar == 'rank':
        T_list = [700]*4
        T0_list = [26]*4
        s_list = [5]*4
        r_list = [2,3,4,5]
        N_list = [20]*4
    elif args.rho == 0.7 and args.dgp_r == 4 and args.changevar == 's':
        s_list = [2,3,4,5,6,7,8,9,10,15,20,25,30,35]
        T_list = [1500]*len(s_list)
        T0_list = [58]*len(s_list)
        r_list = [4]*len(s_list)
        N_list = [20]*len(s_list)


if args.task == 'rate':
    exp_name = f'{args.dgp}_{args.changevar}_true'


# for (p,r,dgp_s) in [(1,1,1),(0,1,1),(0,1,0),(0,0,1)]:
#     for T,T0,s in zip(T_list,T0_list,s_list):
#         for init in ['zero','true','spec']:
#             command = ['python3.9','run.py','--dgp_p',str(p),'--dgp_r',str(r),'--dgp_s',str(dgp_s),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--name',exp_name]
#             print(' '.join(command))
#             subprocess.run(command,stdout=None)

# rate
    for T,T0,s,r,N in zip(T_list,T0_list,s_list,r_list,N_list):
        # for init in ['GD']:
        #     command = ['python3.9','../train.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','100','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task',args.task]
        #     print(' '.join(command))
        #     subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        for init in ['true']:
            command = ['python3.9','../train.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(r),'--rho',str(args.rho),'--n_rep','400','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task',args.task,'--stop_method','Adiff','--schedule','half','--lr','1e-3','--N',str(N),'--seed','100']
            print(' '.join(command))
            subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)

# convergence
elif args.task == 'convergence':
    exp_name = f'{args.dgp}_full_initzero_lr5e-4'
    for T,T0,s in zip(T_list,T0_list,s_list):
        for init in ['zero']:
            command = ['python3.9','../train.py','--dgp',args.dgp,'--season','4','--dgp_p',str(args.dgp_p),'--dgp_r',str(args.dgp_r),'--rho',str(args.rho),'--n_rep','500','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init',init,'--exp_name',exp_name,'--task',args.task,'--schedule','none','--lr','5e-4']
            print(' '.join(command))
            subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)


# season_ar rate
# command = ['python3.9','../train.py','--dgp','season_ar','--season','4','--dgp_p','4','--dgp_r',str(args.dgp_r),'--rho','0.7','--n_rep','500','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init','noisetrue,'--exp_name',exp_name,'--task','rate,'--stop_method','Adiff','--schedule','half','--lr','1e-3']
# arma rate
# command = ['python3.9','../train.py','--dgp','arma','--season','4','--dgp_p','1','--dgp_r',str(args.dgp_r),'--rho','0.7','--n_rep','500','--T',str(T),'--s',str(s),'--T0',str(T0),'--A_init','true','--exp_name',exp_name,'--task','rate','--stop_method','Adiff','--schedule','half','--lr','1e-3']