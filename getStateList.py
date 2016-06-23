import numpy as np
from math import *
from quantumdefects import *


def getMyStates(nState,lState,maxL,QDmargin):
    minhy, maxhy = int(floor(nState-QDmargin*2)), int(ceil(nState+QDmargin*2))
    statelist = np.zeros([(maxL+1)*(maxhy- minhy),2],dtype=np.int)
    i = 0
    for n in range(minhy, maxhy):
        for l in range(0, maxL+1):
            statelist[i,0] = n
            statelist[i,1] = l
            i = i+1

    mystates=np.array([[]],dtype=np.int)
            
    for st in statelist:
        if (abs((nState-deltaQD(nState,lState,lState))-(st[0]-deltaQD(st[0],st[1],st[1]))) < QDmargin and st[1]<st[0]):
            mystates=np.append(mystates,st)

    mystates = np.reshape(mystates,(len(mystates)/2,2))

    mystatesFS = np.array([],dtype=np.int)
    for state in mystates:
        if len(mystatesFS) == 0:
            mystatesFS = np.array([[state[0],state[1],state[1]]],dtype=np.int)
        else:
            mystatesFS = np.append(mystatesFS,[[state[0],state[1],state[1]]],axis=0)
        if state[1] != 0:
            mystatesFS = np.append(mystatesFS,[[state[0],state[1],state[1]-1]],axis=0)

    mystatesFSmj = np.array([],dtype=np.int)
    for state in mystates:
        if len(mystatesFSmj) == 0:
            mystatesFSmj = np.array([[state[0],state[1],state[1],-state[1]-1]],dtype=np.int)
            for mj in range(-state[1],state[1]+1):
                mystatesFSmj = np.append(mystatesFSmj,[[state[0],state[1],state[1],mj]],axis=0)
        else:
            for mj in range(-state[1]-1,state[1]+1):
                mystatesFSmj = np.append(mystatesFSmj,[[state[0],state[1],state[1],mj]],axis=0)
        if state[1] != 0:
            for mj in range(-(state[1]-1)-1,(state[1]-1)+1):
                mystatesFSmj = np.append(mystatesFSmj,[[state[0],state[1],state[1]-1,mj]],axis=0)

    print "Number of states in basis set: {}".format(len(mystatesFS))
    
    return mystates, mystatesFS, mystatesFSmj
        