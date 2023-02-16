"""
Generate N-dimensional time series {y} in AR form given length T, coefficients {Aj} with j from 1 to T
"""
import numpy as np
from help_func import *
from scipy.stats import ortho_group
import matplotlib.pyplot as plt

def DGP_BIC(N,T,burn,p,r,s,r1,r2,P,st_control=2):
    """
    the prob distribution is standard normal
    """
    d = p+r+2*s
    # lmbd,gamma,theta = dgp_rand_w(r,s)
    if r == 5:
        lmbd = [-0.6,-0.4,0.2,0.4,0.6]
    elif r == 4:
        lmbd = [-0.6,-0.3,0.3,0.6]
    elif r == 2:
        lmbd = [-0.5,0.5]
    elif r == 1:
        lmbd = [-0.5]
    elif r == 0:
        lmbd = []
    
    if s == 0:
        gamma = []
        theta = []
    elif s == 1:
        gamma = [0.5]
        theta = [np.pi/4]
    elif s == 2:
        gamma = [0.3,0.6]
        theta = [-np.pi/4,np.pi/4]
    L = get_L(lmbd,gamma,theta,N,r,s,P,p)
    G = generate_G(N,r1,r2,p,r,s,lmbd,gamma,st_control)
    A = mode_dot(G,L,2)
    # while np.sum(np.linalg.norm(A[:,:,:T],axis=(0,1),ord=2)) > 1:
    #     G = generate_G(N,r1,r2,p,r,s,lmbd,gamma,st_control=1)
    #     A = mode_dot(G,L,2)
    y = np.zeros((N,T+burn))
    
    # Norm = 0
    for t in range(T+burn):
        # summand = np.zeros(N)
        y_temp = np.copy(y[:,:t])
        y_temp = np.reshape(np.flip(y_temp,axis=1),(-1,1),order='F')
        L_temp = np.copy(L[:t,:])
        summand = tensor_op.unfold(mode_dot(G,L_temp,2),1) @ y_temp
        # for j in range(t):
        #     for k in range(d):
        #         summand = summand + L[j,k] * G[:,:,k] @ y[:,t-j-1]
        noise = np.random.normal(loc=0.0, scale=0.2, size=N)
        y[:,t] = summand.T[0] + noise
        # Norm = Norm + np.linalg.norm(noise)
        # y[:,t] = summand
    return (y[:,burn:],lmbd,gamma,theta,G,L)

def generate_G(N,r1,r2,p,r,s,lmbd,gamma,st_control=2): # checked
    """
    Generate a 3-D low-Tucker-rank tensor G with size N*N*d and rank r1 in the first mode and rank r2 in the second mode
    """
    d = p+r+2*s
    G = np.random.normal(loc=0,scale=1,size=(N,N,d))
    S,U1,U2 = get_S_and_U(G,r1,r2,d)
    S = normalize_S(S,r1,r2,p,r,s,lmbd,gamma,st_control)
    G = tl.tenalg.mode_dot(S,U1,0)
    G = tl.tenalg.mode_dot(G,U2,1)
    return G

def generate_S(r1,r2,p,r,s,lmbd,gamma,max_iter=20,random_state=None):
    """
    S shoule satisfy the constraints:
    sum_i=1^p S_mat_i + rho/(1-rho) * sum_j=p+1^d S_mat_j < 1
    where rho = max{lambda,gamma}
    """

    for k in range(max_iter):
        d=p+r+2*s
        if (s != 0) & (r != 0):
            rho = np.max([np.max(np.abs(lmbd)),np.max(gamma)])
        elif s != 0:
            rho = np.max(gamma)
        else:
            rho = np.max(np.abs(lmbd))
        S = np.random.randn(r1,r2,d)
        S_fro_norm = np.linalg.norm(S,ord='fro',axis=(0,1))
        S = np.einsum('ijk,k->ijk',S,np.reciprocal(S_fro_norm))
        S = 3*(1-rho)/rho * S / (r1*r2*d)
        S_op_norm = np.linalg.norm(S,ord=2,axis=(0,1))
        summand = np.sum(S_op_norm[:p]) + rho/(1-rho) * np.sum(S_op_norm[p:])
        if summand < 1:
            return S
    print("max_iter exceeded")
    raise ValueError

def normalize_S(S,r1,r2,p,r,s,lmbd,gamma,st_control=2,max_iter=20):
    """
    S shoule satisfy the constraints:
    sum_i=1^p S_mat_i + rho/(1-rho) * sum_j=p+1^d S_mat_j < 1
    where rho = max{lambda,gamma}
    """

    for k in range(max_iter):
        d=p+r+2*s
        if (s != 0) & (r != 0):
            rho = np.max([np.max(np.abs(lmbd)),np.max(gamma)])
        elif s != 0:
            rho = np.max(gamma)
        elif r != 0:
            rho = np.max(np.abs(lmbd))
        else:
            rho = 0.5
        S_fro_norm = np.linalg.norm(S,ord='fro',axis=(0,1))
        S = np.einsum('ijk,k->ijk',S,np.reciprocal(S_fro_norm))
        sigma1 = np.linalg.norm(S[0,:,:],ord='fro')
        S[0,:,:] = S[0,:,:]*np.reciprocal(sigma1)
        sigma2 = np.linalg.norm(S[1,:,:],ord='fro')
        S[1,:,:] = S[1,:,:]*np.reciprocal(sigma2)
        S = 3*st_control*(1-rho)/rho * S / (r1*r2*d)
        S_op_norm = np.linalg.norm(S,ord=2,axis=(0,1))
        summand = np.sum(S_op_norm[:p]) + rho/(1-rho) * np.sum(S_op_norm[p:])
        if summand < st_control:
            return S
    print("max_iter exceeded")
    raise ValueError

def dgp_rand_w(r,s): #checked
    """
    Uniform distribution for now
    (may need to adjust range for endpoint issue) 
    """
    lmbd = np.random.rand(r) *1.8 -0.9 # [-0.9,0.9]
    gamma = np.random.rand(s) * 0.9 # [0,0.9]
    theta = np.random.rand(s)*np.pi - np.pi/2 # [-pi/2,pi/2]
    return (lmbd,gamma,theta)


def DGP_VARMA(N,T,burn,p,r,s):
    # lmbd,gamma,theta = dgp_rand_w(r,s)
    if r == 2:
        lmbd = [-0.8,0.8]
    elif r == 1:
        lmbd = [-0.8]
    elif r == 0:
        lmbd = []
    
    if s == 0:
        gamma = []
        theta = []
    elif s == 1:
        gamma = [0.8]
        theta = [np.pi/4]
    # two coef matrices
    eigenspace = ortho_group.rvs(N)
    if r == 1 and s == 0:
        J = np.array([[-0.8]])
        H = np.array([[0.8]])
    elif r == 1 and s == 1:
        J = np.array([[-0.8,0,0],[0,0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2],[0,-0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2]])
        # J = np.array([[-0.8,0,0],[0,0.8*np.cos(np.pi/8),0.8*np.sin(np.pi/8)],[0,-0.8*np.sin(np.pi/8),0.8*np.cos(np.pi/8)]])
        H = np.array([[0.8,0,0],[0,-0.8,0],[0,0,-0.8]])
    elif r == 0 and s == 1:
        # J = np.array([[0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2],[-0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2]])
        J = np.array([[0.8*np.cos(np.pi/4),0.8*np.sin(np.pi/4)],[-0.8*np.sin(np.pi/4),0.8*np.cos(np.pi/4)]])
        H = np.array([[-0.8,0],[0,-0.8]])

    Theta = eigenspace[:,:r+2*s] @ J @ eigenspace[:,:r+2*s].T
    # Theta1 = eigenspace[:,:r+2*s] @ np.array([[-0.5,0,0],[0,0.5*np.sqrt(2)/2,0.5*np.sqrt(2)/2],[0,-0.5*np.sqrt(2)/2,0.5*np.sqrt(2)/2]]) @ eigenspace[:,:r+2*s].T
    Phi = eigenspace[:,:r+2*s] @ H @ eigenspace[:,:r+2*s].T

    # express as SARMA form
    if p == 1:
        B = eigenspace
        B_minus = B.T @ (Phi-Theta)
        G = np.zeros((N,N,p+r+2*s))
        G[:,:,0] = eigenspace[:,:r+2*s] @ (H-J) @ eigenspace[:,:r+2*s].T
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])
        for i in range(s):
            G[:,:,p+r+2*i] = np.outer(B[:,r+2*i],B_minus[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_minus[r+2*i+1,:])
            G[:,:,p+r+2*i+1] = np.outer(B[:,r+2*i],B_minus[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_minus[r+2*i,:])
        L = get_L(lmbd,gamma,theta,N,r,s,T,p)
        # A = mode_dot(G,L,2)
        # _,st1,_ = np.linalg.svd(tensor_op.unfold(A,1))
        # _,st2,_ = np.linalg.svd(tensor_op.unfold(A,2))
        # print(st1,st2)
        # _,st1,_ = np.linalg.svd(tensor_op.unfold(G,1))
        # _,st2,_ = np.linalg.svd(tensor_op.unfold(G,2))
        # print(st1,st2)
    elif p == 0:
        Phi = np.zeros((N,N))
        B = eigenspace
        B_minus = B.T @ (Phi-Theta)
        G = np.zeros((N,N,p+r+2*s))
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])
        for i in range(s):
            G[:,:,p+r+2*i] = np.outer(B[:,r+2*i],B_minus[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_minus[r+2*i+1,:])
            G[:,:,p+r+2*i+1] = np.outer(B[:,r+2*i],B_minus[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_minus[r+2*i,:])
        L = get_L(lmbd,gamma,theta,N,r,s,T,p)
        _,st1,_ = np.linalg.svd(tensor_op.unfold(G,1))
        _,st2,_ = np.linalg.svd(tensor_op.unfold(G,2))
        print(st1,st2)

    # A = A[:,:,:T]
    # print(np.sum(np.linalg.norm(A,axis=(0,1),ord=2)))
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = Phi @ y[:,t-1] - Theta @ eps[:,t-1] + eps[:,t]

    #plot
    plt.plot(y[0,800:])
    plt.show()
    return (y[:,burn:],lmbd,gamma,theta,G,L)

def DGP_Rank(N,T,burn,p,r,s):
    # lmbd,gamma,theta = dgp_rand_w(r,s)
    if r == 2:
        lmbd = [-0.8,0.8]
    elif r == 1:
        lmbd = [-0.8]
    elif r == 0:
        lmbd = []
    
    if s == 0:
        gamma = []
        theta = []
    elif s == 1:
        gamma = [0.8]
        theta = [np.pi/4]
    # two coef matrices
    eigenspace = ortho_group.rvs(N)
    if r == 1 and s == 0:
        J = np.array([[-0.8,0,0],[0,0,0],[0,0,0]])
        # H = np.array([[0.3,0,0],[0,0.7,0],[0,0,0]]) # for all old simulations
        H = np.array([[0.3,0,0],[0,0.8,0],[0,0,0]]) # for 110 BIC + rank particularly
        # H = np.array([[-0.1,0,0],[0,0.7,0],[0,0,0]])

    elif r == 1 and s == 1:
        J = np.array([[-0.8,0,0],[0,0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2],[0,-0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2]])
        # J = np.array([[-0.8,0,0],[0,0.8*np.cos(np.pi/8),0.8*np.sin(np.pi/8)],[0,-0.8*np.sin(np.pi/8),0.8*np.cos(np.pi/8)]])
        H = np.array([[-0.1,0,0],[0,-0.3,0],[0,0,-0.3]])
        # H = np.array([[-0.3,0,0],[0,-0.3,0],[0,0,-0.3]])
    elif r == 0 and s == 1:
        # J = np.array([[0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2],[-0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2]])
        J = np.array([[0.8*np.cos(np.pi/4),0.8*np.sin(np.pi/4),0],[-0.8*np.sin(np.pi/4),0.8*np.cos(np.pi/4),0],[0,0,0]])
        # J = np.array([[0.8*np.cos(np.pi/8),0.8*np.sin(np.pi/8),0],[-0.8*np.sin(np.pi/8),0.8*np.cos(np.pi/8),0],[0,0,0]])
        # H = np.array([[0.3,0,0],[0,0.2,0],[0,0,0]])
        H = np.array([[-0.8,0,0],[0,-0.8,0],[0,0,0]]) # for 101 good rank
        # H = np.array([[0.2,0,0],[0,0.2,0],[0,0,0.7]])


    Theta = eigenspace[:,:3] @ J @ eigenspace[:,:3].T
    # Theta1 = eigenspace[:,:r+2*s] @ np.array([[-0.5,0,0],[0,0.5*np.sqrt(2)/2,0.5*np.sqrt(2)/2],[0,-0.5*np.sqrt(2)/2,0.5*np.sqrt(2)/2]]) @ eigenspace[:,:r+2*s].T
    Phi = eigenspace[:,:3] @ H @ eigenspace[:,:3].T

    # express as SARMA form
    if p == 1:
        B = eigenspace
        B_minus = B.T @ (Phi-Theta)
        G = np.zeros((N,N,p+r+2*s))
        G[:,:,0] = eigenspace[:,:3] @ (H-J) @ eigenspace[:,:3].T
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])
        for i in range(s):
            G[:,:,p+r+2*i] = np.outer(B[:,r+2*i],B_minus[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_minus[r+2*i+1,:])
            G[:,:,p+r+2*i+1] = np.outer(B[:,r+2*i],B_minus[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_minus[r+2*i,:])
        L = get_L(lmbd,gamma,theta,N,r,s,T,p)
        A = mode_dot(G,L,2)
        _,st1,_ = np.linalg.svd(tensor_op.unfold(A,1))
        _,st2,_ = np.linalg.svd(tensor_op.unfold(A,2))
        print(st1,st2)
        # _,st1,_ = np.linalg.svd(tensor_op.unfold(G,1))
        # _,st2,_ = np.linalg.svd(tensor_op.unfold(G,2))
        # print(st1,st2)
    elif p == 0:
        Phi = np.zeros((N,N))
        B = eigenspace
        B_minus = -B.T
        G = np.zeros((N,N,p+r+2*s))
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])
        for i in range(s):
            G[:,:,p+r+2*i] = np.outer(B[:,r+2*i],B_minus[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_minus[r+2*i+1,:])
            G[:,:,p+r+2*i+1] = np.outer(B[:,r+2*i],B_minus[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_minus[r+2*i,:])
        L = get_L(lmbd,gamma,theta,N,r,s,T,p)
        # _,st1,_ = np.linalg.svd(tensor_op.unfold(G,1))
        # _,st2,_ = np.linalg.svd(tensor_op.unfold(G,2))
        A = mode_dot(G,L,2)
        _,st1,_ = np.linalg.svd(tensor_op.unfold(A,1))
        _,st2,_ = np.linalg.svd(tensor_op.unfold(A,2))
        print(st1,st2)
        # print(st1,st2)

    # A = A[:,:,:T]
    # print(np.sum(np.linalg.norm(A,axis=(0,1),ord=2)))
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = Phi @ y[:,t-1] - Theta @ eps[:,t-1] + eps[:,t]

    #plot
    # plt.plot(y[0,800:])
    # plt.show()
    return (y[:,burn:],lmbd,gamma,theta,G,L)

def DGP_AUTO(N,T,burn,p,r,s,signal):
    # lmbd,gamma,theta = dgp_rand_w(r,s)
    if r == 2:
        lmbd = [-0.8,0.8]
    elif r == 1:
        lmbd = [-signal]
    elif r == 0:
        lmbd = []
    
    if s == 0:
        gamma = []
        theta = []
    elif s == 1:
        gamma = [signal]
        theta = [np.pi/4]
    # two coef matrices
    eigenspace = ortho_group.rvs(N)
    if r == 1 and s == 0:
        J = np.array([[-signal,0,0],[0,0,0],[0,0,0]])
        # H = np.array([[0.3,0,0],[0,0.7,0],[0,0,0]]) # for all old simulations
        H = np.array([[0.3,0,0],[0,0.8,0],[0,0,0]]) # for 110 BIC + rank particularly
        # H = np.array([[-0.1,0,0],[0,0.7,0],[0,0,0]])

    elif r == 1 and s == 1:
        J = np.array([[-signal,0,0],[0,signal*np.sqrt(2)/2,signal*np.sqrt(2)/2],[0,-signal*np.sqrt(2)/2,signal*np.sqrt(2)/2]])
        # J = np.array([[-0.8,0,0],[0,0.8*np.cos(np.pi/8),0.8*np.sin(np.pi/8)],[0,-0.8*np.sin(np.pi/8),0.8*np.cos(np.pi/8)]])
        H = np.array([[-0.1,0,0],[0,-0.3,0],[0,0,-0.3]])
        # H = np.array([[-0.3,0,0],[0,-0.3,0],[0,0,-0.3]])
    elif r == 0 and s == 1:
        # J = np.array([[0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2],[-0.8*np.sqrt(2)/2,0.8*np.sqrt(2)/2]])
        J = np.array([[signal*np.cos(np.pi/4),signal*np.sin(np.pi/4),0],[-signal*np.sin(np.pi/4),signal*np.cos(np.pi/4),0],[0,0,0]])
        # J = np.array([[0.8*np.cos(np.pi/8),0.8*np.sin(np.pi/8),0],[-0.8*np.sin(np.pi/8),0.8*np.cos(np.pi/8),0],[0,0,0]])
        # H = np.array([[0.3,0,0],[0,0.2,0],[0,0,0]])
        H = np.array([[-0.8,0,0],[0,-0.8,0],[0,0,0]]) # for 101 good rank
        # H = np.array([[0.2,0,0],[0,0.2,0],[0,0,0.7]])


    Theta = eigenspace[:,:3] @ J @ eigenspace[:,:3].T
    # Theta1 = eigenspace[:,:r+2*s] @ np.array([[-0.5,0,0],[0,0.5*np.sqrt(2)/2,0.5*np.sqrt(2)/2],[0,-0.5*np.sqrt(2)/2,0.5*np.sqrt(2)/2]]) @ eigenspace[:,:r+2*s].T
    Phi = eigenspace[:,:3] @ H @ eigenspace[:,:3].T

    # express as SARMA form
    if p == 1:
        B = eigenspace
        B_minus = B.T @ (Phi-Theta)
        G = np.zeros((N,N,p+r+2*s))
        G[:,:,0] = eigenspace[:,:3] @ (H-J) @ eigenspace[:,:3].T
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])
        for i in range(s):
            G[:,:,p+r+2*i] = np.outer(B[:,r+2*i],B_minus[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_minus[r+2*i+1,:])
            G[:,:,p+r+2*i+1] = np.outer(B[:,r+2*i],B_minus[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_minus[r+2*i,:])
        L = get_L(lmbd,gamma,theta,N,r,s,T,p)
        A = mode_dot(G,L,2)
        # _,st1,_ = np.linalg.svd(tensor_op.unfold(A,1))
        # _,st2,_ = np.linalg.svd(tensor_op.unfold(A,2))
        # print(st1,st2)
        # _,st1,_ = np.linalg.svd(tensor_op.unfold(G,1))
        # _,st2,_ = np.linalg.svd(tensor_op.unfold(G,2))
        # print(st1,st2)
    elif p == 0:
        Phi = np.zeros((N,N))
        B = eigenspace
        B_minus = -B.T
        G = np.zeros((N,N,p+r+2*s))
        for i in range(r):
            G[:,:,p+i] = np.outer(B[:,i],B_minus[i,:])
        for i in range(s):
            G[:,:,p+r+2*i] = np.outer(B[:,r+2*i],B_minus[r+2*i,:]) + np.outer(B[:,r+2*i+1],B_minus[r+2*i+1,:])
            G[:,:,p+r+2*i+1] = np.outer(B[:,r+2*i],B_minus[r+2*i+1,:]) - np.outer(B[:,r+2*i+1],B_minus[r+2*i,:])
        L = get_L(lmbd,gamma,theta,N,r,s,T,p)
        # _,st1,_ = np.linalg.svd(tensor_op.unfold(G,1))
        # _,st2,_ = np.linalg.svd(tensor_op.unfold(G,2))
        A = mode_dot(G,L,2)
        _,st1,_ = np.linalg.svd(tensor_op.unfold(A,1))
        _,st2,_ = np.linalg.svd(tensor_op.unfold(A,2))
        print(st1,st2)
        # print(st1,st2)

    # A = A[:,:,:T]
    # print(np.sum(np.linalg.norm(A,axis=(0,1),ord=2)))
    y = np.zeros((N,T+burn))
    eps = np.zeros((N,T+burn))
    for t in range(T+burn):
        eps[:,t] = np.random.normal(loc=0.0, scale=1, size=N)
        y[:,t] = Phi @ y[:,t-1] - Theta @ eps[:,t-1] + eps[:,t]

    #plot
    # plt.plot(y[0,800:])
    # plt.show()
    return (y[:,burn:],lmbd,gamma,theta,G,L)

np.random.seed(1)
# DGP_VARMA(10,1000,20,1,1,1)
DGP_AUTO(10,500,200,0,1,0,0.8)
