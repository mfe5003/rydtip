from __future__ import division
import numpy as np

deltaValues = np.array([[0,0,0,0,0,0],[4.04935665,0.2377037,0,0,0,0],[3.59158950,0.360926,0.0,0.0,0,0],[3.5589599,0.392469,0,0,0,0],[2.475365,0.5554,0,0,0,0],[2.46631524,0.013577,0,0,0,0],[0.03341424,-0.198674,0,0,0,0],[0.033537,-0.191,0,0,0,0],[0.00703865*0,-0.049252*0,0,0,0,0],[0.00703865*0,-0.049252*0,0,0,0,0]])
deltaValueswoFS = np.array([[0,0,0,0,0,0],[4.04935665,0.2377037,0.255401,0.00378,0.25486,0],[3.59158950,0.360926,0.41905,0.64388,1.45035,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[2.46631524,0.013577,-0.37457,-2.1867,-1.5532,-56.6739],[0.03341424,-0.198674,0.28953,-0.2601,0,0],[0,0,0,0,0,0],[0.00703865,-0.049252,0.01291,0,0,0],[0,0,0,0,0,0]])
deltaValuesRb = np.array([[0,0,0,0,0,0],[3.1311804,0.1784,0,0,0,0],[2.6548849,0.29,0,0,0,0],[2.6416737,0.295,0,0,0,0],[1.34809171,-0.60286,0,0,0,0],[1.34646572,-0.596,0,0,0,0],[0.0165912,-0.085,0,0,0,0],[0.0165437,-0.086,0,0,0,0],[0.00405,0,0,0,0,0],[0.00405,0,0,0,0,0]])

def setupFS(Ry):
        
    SpeedOfLight = 299792458
    MHztoInvCmToHartree = 1e6/(SpeedOfLight*1e2)/(2*Ry)

    Amat = np.array(map(lambda x: map(lambda y: y*MHztoInvCmToHartree, x), [[2.13925e8, -5.6e7, 3.9e8],[6.02183e7, -5.8e7, 0], [-9.796e5, 1.222e7, -3.376e7]]))
    einf = np.array([3.57531,2.47079,0.03346])
    ainf = np.array([0.3727,0.0612,-0.191])
    
    return Amat, einf, ainf

def epsilonFS(n,l,j):
    Ry = 109736.86224
    Amat, einf, ainf = setupFS(Ry)
    return einf[l-1]+ainf[l-1]*(n-einf[l-1])**-2

def DeltaFS(n,l,j):
    Ry = 109736.86224
    Amat, einf, ainf = setupFS(Ry)
    if (l==1 or l==2 or l==3):
        return Amat[l-1,0]*(n-epsilonFS(n,l,j))**-3 +  \
                Amat[l-1,1]*(n-epsilonFS(n,l,j))**-5 + \
                Amat[l-1,2]*(n-epsilonFS(n,l,j))**-7
    else:
        return 0

def deltaK(k,l,j):
    if (2*l+2 <= len(deltaValues)):
        return deltaValues[l+1+j,k/2]
    else:
        return 0
        
def deltaKRb(k,l,j):
    if (2*l+2 <= len(deltaValuesRb)):
        return deltaValuesRb[l+1+j,k/2]
    else:
        return 0
        
def deltaKwoFS(k,l,j):
    if (2*l+2 <= len(deltaValueswoFS)):
        return deltaValueswoFS[l+1+j,k/2]
    else:
        return 0
        
def deltaQD(n,l,j):
    sum=0
    np.seterr(all='raise')
    #print "{} {} {}".format(n,l,j)
    for k in range(1,len(deltaValues[0])-1):
        if (deltaK(0,l,j)!=0):
            sum += deltaK(2*k,l,j)/(n-deltaK(0,l,j))**(2*k)
    qd=deltaK(0,l,j)+sum
    return qd
    
def deltaQDRb(n,l,j):
    sum=0
    for k in range(1,len(deltaValuesRb[0])-1):
        if (deltaKRb(0,l,j) != 0):
            sum += deltaKRb(2*k,l,j)/(n-deltaKRb(0,l,j))**(2*k)
    qd=deltaKRb(0,l,j)+sum
    return qd
    
def deltaQDwoFS(n,l,j):
    sum=0
    for k in range(1,len(deltaValueswoFS[0])-1):
        if (deltaKwoFS(0,l,j) != 0):
            sum += deltaKwoFS(2*k,l,j)/(n-deltaKwoFS(0,l,j))**(2*k)
    qd=deltaKwoFS(0,l,j)+sum
    return qd

def zeroFieldEnergy(n,l,j,z):
    if z == 37:
        return -1.0/(2*(n-deltaQDRb(n,l,j))**2)
    elif z == 55:
        if (l==0 and j==0):
            return -1.0/(2*(n-deltaQDwoFS(n,l,j))**2)
        elif (l==1 and j==0):
            return -1.0/(2*(n-deltaQDwoFS(n,l,j))**2)
        elif (l==1 and j==1):
            return -1.0/(2*(n-deltaQDwoFS(n,l,j-1))**2) + DeltaFS(n,l,j-1)
        elif (l==2 and j==1):
            return -1.0/(2*(n-deltaQDwoFS(n,l,j+1))**2) - DeltaFS(n,l,j)
        elif (l==2 and j==2):
            return -1.0/(2*(n-deltaQDwoFS(n,l,j))**2)
        elif (l==3 and j==2):
            return -1.0/(2*(n-deltaQDwoFS(n,l,j))**2)
        elif (l==3 and j==3):
            return -1.0/(2*(n-deltaQDwoFS(n,l,j-1))**2) + DeltaFS(n,l,j-1)
        elif (l==4 and j==3):
            return -1.0/(2*(n-deltaQDwoFS(n,l,j))**2)
        elif (l==4 and j==4):
            return -1.0/(2*(n-deltaQDwoFS(n,l,j-1))**2) + DeltaFS(n,l,j-1)
        else:
            return -1.0/(2*(n-deltaQDwoFS(n,l,j))**2)