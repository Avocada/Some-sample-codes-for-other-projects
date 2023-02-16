"""
Model parameters:
p: AR part order
q: MA part order
r: no. of real eigenvalues
s: no. of complex eigenvalue pairs
d = r+2s
r1: rank of first mode of A
r2: rank of second mode of A
N: dimension of the time series vector
T: length of the time series

Variables:
A: N*N*inf tensor
G: N*N*d tensor, G = S*U1*U2
S: r1*r2*d tensor
U1: N*r1 matrix
U2: N*r2 matrix
L: inf*d matrix
w: d*1 vector
lmbd: r*1 vector
gamma: s*1 vector
theta: s*1 vector

Data:
y: N*T matrix, stored as column vectors, from old (left) to new (right)
"""

from math import sqrt
import numpy as np
import scipy
from tensorOp import tensor_op
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tenalg import kronecker as kron
from tensorly.tenalg import mode_dot
import matplotlib.pyplot as plt

##################
# Initialization #
##################

def init_A(y,N,T,P,r1,r2):
    """
    Use OLS method to initialize the tensor A
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A
    """
    # create X (regressors)
    X = np.zeros((N*P,T-P))
    for i in range(P):
        X[i*N:i*N+N,:] = y[:,P-i-1:T-i-1]
    # create Y (response)
    Y = y[:,P:]
    # solve OLS
    A = (X@Y.T).T @ np.linalg.inv(X @ X.T)
    # fold A into a tensor
    A = np.array(tensor_op.fold(A,(N,N,P),1))
    # HOOI to get a low rank version
    A,U = tucker(A,rank=[r1,r2,P])
    A = tl.tenalg.multi_mode_dot(A,U)
    return A

def spectral_init_A(y,N,T,P,r1,r2):
    """
    Spectral initialization
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A
    """
    # create X (regressors)
    X = np.zeros((N*P,T-P))
    for i in range(P):
        X[i*N:i*N+N,:] = y[:,P-i-1:T-i-1]
    # create Y (response)
    Y = y[:,P:]
    # spectral initialization
    A = np.zeros((N,N*P))
    for t in range(T-P):
        A = A + np.outer(Y[:,t],X[:,t])
    A = A/(T-P)
    # fold A into a tensor
    A = np.array(tensor_op.fold(A,(N,N,P),1))

    # HOOI to get a low rank version
    A,U = tucker(A,rank=[r1,r2,P])
    A = tl.tenalg.multi_mode_dot(A,U)
    return A

def spectral_init_A_exp(y,N,T,P,r1,r2):
    """
    Spectral initialization by Zhang AR
    y: N*T array
    Y: N*P (truncated) array
    X: NP*(T-P) array
    return A
    """
    # create X (regressors)
    X = np.zeros((N*P,T-P))
    for i in range(P):
        X[i*N:i*N+N,:] = y[:,P-i-1:T-i-1]
    # create Y (response)
    Y = y[:,P:]
    # spectral initialization
    A = np.zeros((N,N*P))
    for t in range(T-P):
        A = A + np.outer(Y[:,t],X[:,t])
    A = A/(T-P)
    # fold A into a tensor
    A = np.array(tensor_op.fold(A,(N,N,P),1))

    # HOOI to get a low rank version
    S,U = tucker(A,rank=[r1,r2,P])
    S = mode_dot(S,U[2],2)
    return S,U[0],U[1]

def rand_w(r,s): #checked
    """
    Uniform distribution for now
    (may need to adjust range for endpoint issue)
    """
    lmbd = np.random.rand(r) *2 -1 # [-1,1]
    gamma = np.random.rand(s) # [0,1]
    theta = np.random.rand(s)*np.pi - np.pi/2 # [-pi/2,pi/2]
    return (lmbd,gamma,theta)

def get_L_MA(lmbd,gamma,theta,N,r,s,P): # checked
    """
    Compute the L_MA matrix given the parameters
    Set size to be P (truncated)
    """
    L = np.zeros((P,r+2*s))
    tri_series = np.zeros((P,2*s))
    for i in range(P):
        tri_series[i,::2] = np.cos((i+1)*np.array(theta))
        tri_series[i,1::2] = np.sin((i+1)*np.array(theta))
        for j in range(r):
            L[i,j] = np.power(lmbd[j],i+1)
        for j in range(s):
            L[i,r+2*j:r+2*j+2] = np.power(gamma[j],i+1)
    new = np.concatenate([L[:,:r], np.einsum('ij,ij -> ij',L[:,r:],tri_series)],axis=1)
    return new

def get_L(lmbd,gamma,theta,N,r,s,P,p): # checked
    """
    Compute the L matrix given the parameters
    Set size to be P (truncated)
    """
    L_MA = get_L_MA(lmbd,gamma,theta,N,r,s,P-p)
    L = np.zeros((P,p+r+2*s))
    L[:p,:p] = np.identity(p)
    L[p:,p:] = L_MA
    return L

def get_G(A,L):
    """
    Restore G from A and L
    G = A inv(L'L)L'
    """
    factor = np.matmul(np.linalg.pinv(np.matmul(L.T,L)),L.T) 
    G = mode_dot(A,factor,2)
    return G

def get_S_and_U(G,r1,r2,d): # checked
    """
    Use HOOI to get S, U1 and U2 from G
    """
    S,U = tucker(G,rank=[r1,r2,d])
    S = mode_dot(S,U[2],2)
    return (S,U[0],U[1])


##################
# Tool Functions #
##################

def loss(y,G,L,T):
    summand = 0
    for t in range(1,T): # starting from y_2
        y_temp = np.copy(y[:,:t])
        y_temp = np.flip(y_temp,axis=1)
        y_temp = np.reshape(y_temp,(-1,1),order='F')
        L_temp = np.copy(L[:t,:])
        a = (tensor_op.unfold(mode_dot(G,L_temp,2),1) @ y_temp).T[0]  # check later
        # a = 0
        # for j in range(t):
        #     for k in range(5):
        #         a = a + L[j,k] * G[:,:,k] @ y[:,t-j]
        summand = summand + np.linalg.norm(y[:,t]-np.array(a),ord=2)**2
    return summand/T

def loss_vec(Y,X1,G,L,T):
    A = mode_dot(G,L,2)
    return sum(np.linalg.norm(Y - tensor_op.unfold(A[:,:,:T-1],1).numpy()@X1,ord=2,axis=0)**2)/T


###############
# Derivatives #
###############

"""
Prepare calleble objective, Jacobian and Hessian functions
"""

def obj_lmbd(lmbd_k,k,G,L,y,p,T):
    """
    Calculate the value of the objective function
    """
    summand = 0
    for t in range(1,T): # starting from y_2
        y_temp = y[:,:t]
        y_temp = np.reshape(np.flip(y_temp,axis=1),(-1,1),order='F')
        L_temp = np.copy(L[:t,:])
        L_temp[:,p+k] = 0
        a = (tensor_op.unfold(mode_dot(G,L_temp,2),1) @ y_temp).T[0] # check later
        power_series = np.arange(1,t-p+1)
        lmbd_power = np.power(lmbd_k,power_series)
        y_temp = y[:,:t-p]
        y_temp = np.flip(y_temp,axis=1)
        b = G[:,:,k] @ np.einsum('i,ji->j',lmbd_power,y_temp)
        summand = summand + np.linalg.norm(y[:,t]-a-b,ord=2)
    return summand


def jac_lmbd(lmbd_k,k,G,L,y,p,T):
    """
    Calculate the gradient of lmbd_k
    """
    summand = 0
    for t in range(1,T):
        y_temp = y[:,:t]
        y_temp = np.reshape(np.flip(y_temp,axis=1),(-1,1),order='F') # vectorized first t-1 y's
        L_temp = np.copy(L[:t,:])
        L_temp[:,p+k] = 0
        # G_temp = G[:]
        # G_temp[:,:,k] = 0
        a = y[:,t] - np.array(tensor_op.unfold(mode_dot(G,L_temp,2),1) @ y_temp).T[0] # check later
        power_series = np.arange(1,t-p+1)
        lmbd_power = np.power(lmbd_k,power_series)
        y_temp = y[:,:t-p]
        y_temp = np.flip(y_temp,axis=1)
        lmbd_y = np.einsum('i,ji->ji',lmbd_power,y_temp)
        outer_grad = a - G[:,:,k] @ np.sum(lmbd_y,axis=1)
        inner_grad = -G[:,:,k] @ np.einsum('i,ji->j',power_series,(lmbd_y/lmbd_k))
        summand = summand + 2*outer_grad @ inner_grad
    return summand/T
    
def hess_lmbd(lmbd_k,k,G,L,y,p,T):
    """
    Calculate the hessian of lmbd_k
    """
    summand = 0
    for t in range(1,T):
        y_temp = y[:,:t]
        y_temp = np.reshape(np.flip(y_temp,axis=1),(-1,1),order='F')
        L_temp = np.copy(L[:t,:])
        L_temp[:,p+k] = 0
        a = y[:,t] - (tensor_op.unfold(mode_dot(G,L_temp,2),1).numpy() @ y_temp).T[0] # check later
        power_series = np.arange(1,t-p+1)
        lmbd_power = np.power(lmbd_k,power_series)
        y_temp = y[:,:t-p]
        y_temp = np.flip(y_temp,axis=1)
        lmbd_y = np.einsum('i,ji->ji',lmbd_power,y_temp)
        outer_grad = a - G[:,:,k] @ np.sum(lmbd_y,axis=1)
        inner_grad = -G[:,:,k] @ np.einsum('i,ji->j',power_series,(lmbd_y/lmbd_k))
        second_grad = -G[:,:,k] @ np.einsum('i,ji->j',power_series[:-1],lmbd_y[:,1:]/lmbd_k)
        summand = summand + 2* outer_grad @ second_grad + 2*inner_grad @ inner_grad
    return summand/T

def jac_hess_lmbd(lmbd_k,k,G,L,y,p,T):
    """
    Calculate the hessian of lmbd_k
    """
    summand_j = 0
    summand_h = 0
    for t in range(1,T):
        y_temp = y[:,:t]
        y_temp = np.reshape(np.flip(y_temp,axis=1),(-1,1),order='F')
        L_temp = np.copy(L[:t,:])
        L_temp[:,p+k] = 0
        a = y[:,t] - (tensor_op.unfold(mode_dot(G,L_temp,2),1).numpy() @ y_temp).T[0] # check later
        power_series = np.arange(1,t-p+1)
        lmbd_power = np.power(lmbd_k,power_series)
        y_temp = y[:,:t-p]
        y_temp = np.flip(y_temp,axis=1)
        lmbd_y = np.einsum('i,ji->ji',lmbd_power,y_temp)
        outer_grad = a - G[:,:,k] @ np.sum(lmbd_y,axis=1)
        inner_grad = -G[:,:,k] @ np.einsum('i,ji->j',power_series,(lmbd_y/lmbd_k))
        second_grad = -G[:,:,k] @ np.einsum('i,ji->j',power_series[:-1],lmbd_y[:,1:]/lmbd_k)
        summand_j = summand_j + 2*outer_grad @ inner_grad
        summand_h = summand_h + 2* outer_grad @ second_grad + 2*inner_grad @ inner_grad
    return summand_j/T,summand_h/T

def vec_jac_hess_lmbd(lmbd_k,k,G,L,Y,X1,X2,p,T):
    """
    vectorization 
    assume already have 
    Y = [y1,...,yT] and 
    X1 = [y[:,:1],...,y[:,:T]] and
    X2 = 
    """
    L_temp = np.copy(L[:(T-1),:])
    L_temp[:,p+k] = 0
    a = Y - (tensor_op.unfold(mode_dot(G,L_temp,2),1).numpy() @ X1) # N by (T-1)

    power_series= np.arange(1,T)
    lmbd_power = np.power(lmbd_k,power_series)
    lmbd_y = np.einsum('i,jik->jik',lmbd_power,X2)
    outer_grad = a - G[:,:,p+k] @ np.sum(lmbd_y,axis=1).T
    inner_grad = -G[:,:,p+k] @ np.einsum('i,jik->jk',power_series,(lmbd_y/lmbd_k)).T
    second_grad = -G[:,:,p+k] @ np.einsum('i,jik->jk',power_series[:-1],lmbd_y[:,1:]/lmbd_k).T
    summand_j = 2* np.einsum('ij,ij->',outer_grad,inner_grad)
    summand_h = 2* np.einsum('ij,ij->',outer_grad,second_grad) + 2* np.einsum('ij,ij->',inner_grad,inner_grad)
    return summand_j/T,summand_h/T


def obj_gamma_theta(eta_k,k,G,L,y,p,r,T):
    gamma_k = eta_k[0]
    theta_k = eta_k[1]
    summand = 0
    for t in range(1,T):
        y_temp = y[:,:t]
        y_temp = np.reshape(np.flip(y_temp,axis=1),(-1,1),order='F')
        L_temp = np.copy(L[:t,:])
        L_temp[:,p+r+k:p+r+k+2] = 0 # set gamma_k and theta_k = 0 in L
        a = y[:,t] - (tensor_op.unfold(mode_dot(G,L_temp,2),1) @ y_temp).T[0] # check later
        power_series = np.arange(1,t-p+1)
        gamma_power = np.power(gamma_k,power_series)
        y_temp = y[:,:t-p]
        y_temp = np.flip(y_temp,axis=1)
        cosine_part = np.einsum('i,ji,i->ji',np.cos(theta_k*power_series), y_temp, gamma_power) #gamma^(j-p) cos{(j-p)theta} y_{t-j}
        sine_part = np.einsum('i,ji,i->ji',np.sin(theta_k*power_series), y_temp, gamma_power) #gamma^(j-p) sin{(j-p)theta} y_{t-j}
        A = G[:,:,p+r+2*k]
        B = G[:,:,p+r+2*k+1]
        summand = summand + np.linalg.norm(a - A @ np.sum(cosine_part,axis=1) - B @ np.sum(sine_part,axis=1),ord=2)
    return summand/T


def jac_gamma_theta(eta_k,k,G,L,y,p,r,T):
    """
    Calculate the gradient of (gamma,theta) pair
    """
    gamma_k = eta_k[0]
    theta_k = eta_k[1]
    summand_gamma = 0
    summand_theta = 0
    for t in range(1,T):
        y_temp = y[:,:t]
        y_temp = np.reshape(np.flip(y_temp,axis=1),(-1,1),order='F')
        L_temp = np.copy(L[:t,:])
        L_temp[:,p+r+k:p+r+k+2] = 0 # set gamma_k and theta_k = 0 in L
        a = y[:,t] - (tensor_op.unfold(mode_dot(G,L_temp,2),1).numpy() @ y_temp).T[0] # check later
        power_series = np.arange(1,t-p+1)
        gamma_power = np.power(gamma_k,power_series)
        y_temp = y[:,:t-p]
        y_temp = np.flip(y_temp,axis=1)
        cosine_part = np.einsum('i,ji,i->ji',np.cos(theta_k*power_series), y_temp, gamma_power) #gamma^(j-p) cos{(j-p)theta} y_{t-j}
        sine_part = np.einsum('i,ji,i->ji',np.sin(theta_k*power_series), y_temp, gamma_power) #gamma^(j-p) sin{(j-p)theta} y_{t-j}
        A = G[:,:,p+r+2*k]
        B = G[:,:,p+r+2*k+1]
        outer_grad = a - A @ np.sum(cosine_part,axis=1) - B @ np.sum(sine_part,axis=1)
        inner_grad_gamma = -A @ np.einsum('i,ji->j',power_series,(cosine_part/gamma_k)) - B @ np.einsum('i,ji->j',power_series,(sine_part/gamma_k))
        inner_grad_theta = A @ np.einsum('i,ji->j',power_series,(sine_part)) - B @ np.einsum('i,ji->j',power_series,(cosine_part))
        summand_gamma = summand_gamma + 2*outer_grad @ inner_grad_gamma
        summand_theta = summand_theta + 2*outer_grad @ inner_grad_theta
    return summand_gamma/T,summand_theta/T

def jac_hess_gamma_theta(eta_k,k,G,L,y,p,r,T):
    """ 
    Calculate the hessian matrix of (gamma,theta) pair
    """
    gamma_k = eta_k[0]
    theta_k = eta_k[1]
    summand_gamma = 0
    summand_theta = 0
    summand_gg = 0
    summand_gt = 0
    summand_tt = 0
    for t in range(1,T):
        y_temp = y[:,:t]
        y_temp = np.reshape(np.flip(y_temp,axis=1),(-1,1),order='F')
        L_temp = np.copy(L[:t,:])
        L_temp[:,p+r+k:p+r+k+2] = 0 # set gamma_k and theta_k = 0 in L
        a = y[:,t] - (tensor_op.unfold(mode_dot(G,L_temp,2),1).numpy() @ y_temp).T[0] # check later
        power_series = np.arange(1,t-p+1)
        gamma_power = np.power(gamma_k,power_series)
        y_temp = y[:,:t-p]
        y_temp = np.flip(y_temp,axis=1)
        cos_part = np.einsum('i,ji,i->ji',np.cos(theta_k*power_series), y_temp, gamma_power) #gamma^(j-p) cos{(j-p)theta} y_{t-j}
        sin_part = np.einsum('i,ji,i->ji',np.sin(theta_k*power_series), y_temp, gamma_power) #gamma^(j-p) sin{(j-p)theta} y_{t-j}
        cos_part_1 = np.einsum('i,ji->ji',power_series,(cos_part/gamma_k)) # (j-p) gamma^(j-p-1) cos{(j-p)theta} y_{t-j}
        sin_part_1 = np.einsum('i,ji->ji',power_series,(sin_part/gamma_k)) # (j-p) gamma^(j-p-1) sin{(j-p)theta} y_{t-j}
        cos_part_2 = cos_part_1[:,1:]/gamma_k # (j-p) gamma^(j-p-2) cos{(j-p)theta} y_{t-j}
        sin_part_2 = sin_part_1[:,1:]/gamma_k # (j-p) gamma^(j-p-2) sin{(j-p)theta} y_{t-j}
        A = G[:,:,p+r+2*k] 
        B = G[:,:,p+r+2*k+1]
        outer_grad = a - A @ np.sum(cos_part,axis=1) - B @ np.sum(sin_part,axis=1)
        inner_grad_gamma = -A @ np.einsum('i,ji->j',power_series,(cos_part/gamma_k)) - B @ np.einsum('i,ji->j',power_series,(sin_part/gamma_k))
        inner_grad_theta = A @ np.einsum('i,ji->j',power_series,(sin_part)) - B @ np.einsum('i,ji->j',power_series,(cos_part))
        summand_gamma = summand_gamma + 2*outer_grad @ inner_grad_gamma
        summand_theta = summand_theta + 2*outer_grad @ inner_grad_theta

        summand_gg = summand_gg + 2 * inner_grad_gamma @ inner_grad_gamma + 2 * outer_grad @ (-A @ np.einsum('i,ji->j',power_series[:-1],cos_part_2) - B @ np.einsum('i,ji->j',power_series[:-1],sin_part_2))
        summand_gt = summand_gt + 2 * inner_grad_gamma @ inner_grad_theta + 2 * outer_grad @ (A @ np.einsum('i,ji->j',power_series,sin_part_1) - B @ np.einsum('i,ji->j',power_series,cos_part_1))
        summand_tt = summand_tt + 2 * inner_grad_theta @ inner_grad_theta + 2 * outer_grad @ (A @ np.einsum('i,i,ji->j',power_series,power_series,cos_part) + B @ np.einsum('i,i,ji->j',power_series,power_series,sin_part))
    return summand_gamma/T,summand_theta/T,np.array([[summand_gg,summand_gt],[summand_gt,summand_tt]])/T

def hess_gamma_theta(eta_k,k,G,L,y,p,r,T):
    """ 
    Calculate the hessian matrix of (gamma,theta) pair
    """
    gamma_k = eta_k[0]
    theta_k = eta_k[1]
    summand_gg = 0
    summand_gt = 0
    summand_tt = 0
    for t in range(1,T):
        y_temp = y[:,:t]
        y_temp = np.reshape(np.flip(y_temp,axis=1),(-1,1),order='F')
        L_temp = np.copy(L[:t,:])
        L_temp[:,p+r+k:p+r+k+2] = 0 # set gamma_k and theta_k = 0 in L
        a = y[:,t] - (tensor_op.unfold(mode_dot(G,L_temp,2),1).numpy() @ y_temp).T[0] # check later
        power_series = np.arange(1,t-p+1)
        gamma_power = np.power(gamma_k,power_series)
        y_temp = y[:,:t-p]
        y_temp = np.flip(y_temp,axis=1)
        cos_part = np.einsum('i,ji,i->ji',np.cos(theta_k*power_series), y_temp, gamma_power) #gamma^(j-p) cos{(j-p)theta} y_{t-j}
        sin_part = np.einsum('i,ji,i->ji',np.sin(theta_k*power_series), y_temp, gamma_power) #gamma^(j-p) sin{(j-p)theta} y_{t-j}
        cos_part_1 = np.einsum('i,ji->ji',power_series,(cos_part/gamma_k)) # (j-p) gamma^(j-p-1) cos{(j-p)theta} y_{t-j}
        sin_part_1 = np.einsum('i,ji->ji',power_series,(sin_part/gamma_k)) # (j-p) gamma^(j-p-1) sin{(j-p)theta} y_{t-j}
        cos_part_2 = cos_part_1[:,1:]/gamma_k # (j-p) gamma^(j-p-2) cos{(j-p)theta} y_{t-j}
        sin_part_2 = sin_part_1[:,1:]/gamma_k # (j-p) gamma^(j-p-2) sin{(j-p)theta} y_{t-j}
        A = G[:,:,p+r+2*k] 
        B = G[:,:,p+r+2*k+1]
        outer_grad = a - A @ np.sum(cos_part,axis=1) - B @ np.sum(sin_part,axis=1)
        inner_grad_gamma = -A @ np.einsum('i,ji->j',power_series,(cos_part/gamma_k)) - B @ np.einsum('i,ji->j',power_series,(sin_part/gamma_k))
        inner_grad_theta = A @ np.einsum('i,ji->j',power_series,(sin_part)) - B @ np.einsum('i,ji->j',power_series,(cos_part))
        
        summand_gg = summand_gg + 2 * inner_grad_gamma @ inner_grad_gamma + 2 * outer_grad @ (-A @ np.einsum('i,ji->j',power_series[:-1],cos_part_2) - B @ np.einsum('i,ji->j',power_series[:-1],sin_part_2))
        summand_gt = summand_gt + 2 * inner_grad_gamma @ inner_grad_theta + 2 * outer_grad @ (A @ np.einsum('i,ji->j',power_series,sin_part_1) - B @ np.einsum('i,ji->j',power_series,cos_part_1))
        summand_tt = summand_tt + 2 * inner_grad_theta @ inner_grad_theta + 2 * outer_grad @ (A @ np.einsum('i,i,ji->j',power_series,power_series,cos_part) + B @ np.einsum('i,i,ji->j',power_series,power_series,sin_part))
    return np.array([[summand_gg,summand_gt],[summand_gt,summand_tt]])/T

def vec_jac_hess_gamma_theta(eta_k,k,G,L,Y,X1,X2,p,r,T):
    """ 
    Calculate the hessian matrix of (gamma,theta) pair
    """
    gamma_k = eta_k[0]
    theta_k = eta_k[1]

    L_temp = np.copy(L[:(T-1),:])
    L_temp[:,p+r+k:p+r+k+2] = 0 # set gamma_k and theta_k = 0 in L
    a = Y - (tensor_op.unfold(mode_dot(G,L_temp,2),1).numpy() @ X1) # check later
    power_series = np.arange(1,T)
    gamma_power = np.power(gamma_k,power_series)
    
    cos_part = np.einsum('i,jik,i->jik',np.cos(theta_k*power_series), X2, gamma_power) #gamma^(j-p) cos{(j-p)theta} y_{t-j}
    sin_part = np.einsum('i,jik,i->jik',np.sin(theta_k*power_series), X2, gamma_power) #gamma^(j-p) sin{(j-p)theta} y_{t-j}
    cos_part_1 = np.einsum('i,jik->jik',power_series,(cos_part/gamma_k)) # (j-p) gamma^(j-p-1) cos{(j-p)theta} y_{t-j}
    sin_part_1 = np.einsum('i,jik->jik',power_series,(sin_part/gamma_k)) # (j-p) gamma^(j-p-1) sin{(j-p)theta} y_{t-j}
    cos_part_2 = cos_part_1[:,1:,:]/gamma_k # (j-p) gamma^(j-p-2) cos{(j-p)theta} y_{t-j}
    sin_part_2 = sin_part_1[:,1:,:]/gamma_k # (j-p) gamma^(j-p-2) sin{(j-p)theta} y_{t-j}
    A = G[:,:,p+r+2*k] 
    B = G[:,:,p+r+2*k+1]
    outer_grad = a - A @ np.sum(cos_part,axis=1).T - B @ np.sum(sin_part,axis=1).T
    inner_grad_gamma = -A @ np.einsum('i,jik->jk',power_series,(cos_part/gamma_k)).T - B @ np.einsum('i,jik->jk',power_series,(sin_part/gamma_k)).T
    inner_grad_theta = A @ np.einsum('i,jik->jk',power_series,(sin_part)).T - B @ np.einsum('i,jik->jk',power_series,(cos_part)).T
    summand_gamma = 2* np.einsum('ij,ij->',outer_grad,inner_grad_gamma)
    summand_theta = 2* np.einsum('ij,ij->',outer_grad,inner_grad_theta)

    summand_gg = 2 * np.einsum('ij,ij->',inner_grad_gamma,inner_grad_gamma) + 2 * np.einsum('ij,ij->', outer_grad, (-A @ np.einsum('i,jik->jk',power_series[:-1],cos_part_2).T - B @ np.einsum('i,jik->jk',power_series[:-1],sin_part_2).T))
    summand_gt = 2 * np.einsum('ij,ij->',inner_grad_gamma,inner_grad_theta) + 2 * np.einsum('ij,ij->', outer_grad, (A @ np.einsum('i,jik->jk',power_series,sin_part_1).T - B @ np.einsum('i,jik->jk',power_series,cos_part_1).T))
    summand_tt = 2 * np.einsum('ij,ij->',inner_grad_theta,inner_grad_theta) + 2 * np.einsum('ij,ij->', outer_grad, (A @ np.einsum('i,i,jik->jk',power_series,power_series,cos_part).T + B @ np.einsum('i,i,jik->jk',power_series,power_series,sin_part).T))
    return summand_gamma/T,summand_theta/T,np.array([[summand_gg,summand_gt],[summand_gt,summand_tt]])/T


#####################
# Useful Components #
#####################

"""
Compute components used in the ALS algorithm
z:
Z:
G_mat:
w_minus: 
oo
"""

def get_z(x,N,L):
    # !!!! need to check dimensions
    z = kron([L.T,np.identity(N)]) @ x
    return z

def get_sample_size(N,p,r,s):
    ratio = np.array([0.4,0.3,0.2,0.1])
    d = r+2*s
    NofP = 4*N + 4*(p+d) + d
    ss = NofP/ratio
    return np.array(np.round(ss),dtype=int)