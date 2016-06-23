from __future__ import division
from scipy import floor, sqrt
from scipy.misc import factorial
from math import *
import numpy as np

def nu1(j,mp,m):
    return min(j-mp,j+m)
    
def nu0(j,mp,m):
    return max(-(mp-m),0)
    
def drot(j,mp,m,theta):
    fac = sqrt(factorial(j+m)*factorial(j-m)*factorial(j+mp)*factorial(j-mp))
    sumterm = 0
    if (nu0(j,mp,m)<=nu1(j,mp,m)):
        for nu in np.arange(nu0(j,mp,m),nu1(j,mp,m)+1):
            if (abs(mp-m+2*nu) < 1e-9):
                sinterm = 1
            else:
                sinterm = pow(-sin(theta/2),mp-m+2*nu)
            sumterm += pow(-1,nu)*pow(cos(theta/2),2*j+m-mp-2*nu)/(factorial(j-mp-nu)*factorial(j+m-nu)*factorial(nu+mp-m)*factorial(nu))*sinterm
            
    return fac*sumterm
            
def ddrot(j,mp,m,phi,theta,chi):
    if (abs(mp)>j or abs(m)>j):
        return 0
    else:
        #return exp(-1j*mp*phi)*drot(j,mp,m,theta)*exp(-1j*m*chi)
        return drot(j,mp,m,theta)                                   #This version does not rotate in phi or chi.

def rotmat(theta,mystatesFS):
    mat = np.zeros((len(mystatesFS),len(mystatesFS)),dtype=np.complex)
    for n,st1 in enumerate(mystatesFS):
        for m,st2 in enumerate(mystatesFS):
            if (st1[0]==st2[0] and st1[1]==st2[1] and st1[2]==st2[2]):
                mat[n,m] = ddrot(max(st1[2],st2[2])+0.5,st2[3]+0.5,st1[3]+0.5,0,theta,0)
    return  mat 