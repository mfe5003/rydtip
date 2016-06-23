#!/usr/bin/python

from __future__ import division
import os
import sys
import numpy as np
import scipy as sp
from scipy.special import *
from math import *
import time
try:
    from sympy.mpmath import *
except:
    from mpmath import *

import matplotlib.pyplot as plt

from wignerrotation import rotmat
from clebsch import clebsch
from quantumdefects import *
from getStateList import *
from radialmatrixelements import rme
from utilityfunctions import *

if (len(sys.argv) != 8):
    print "Incorrect number of command line arguments."
    print ""
    print "Program usage:"
    print "{} n l acf w theta emin emax estep cluster floquetterms vecnum".format(sys.argv[0])
    print ""
    print "n: Principal quantum number of state of interest"
    print "l: Orbital quantum number of state of interest"
    print "emin, emax, estep: min, max, and step for DC electric field (V/cm)"
    print "cluster: identifier for the job which this process belongs to."
    print "vecnum: E-field step number for which to export eigenvectors. Use -1 to not export any eigenvectors."
    print ""
    print "Example usage:"
    print "{} 90 1 0 0.05 0.001 8675309 -1".format(sys.argv[0])
    quit()
   
print "Start time: {}".format(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
   
basedir=""
nState=int(sys.argv[1])
lState=int(sys.argv[2])

eminvcm, emaxvcm, estepvcm = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

cluster=sys.argv[6]

vecnum=int(float(sys.argv[7]))
if (vecnum < 0):
    exportVecs=False
else:
    exportVecs=True


outputunit = 29979

atomicnumber = 55

maxL = nState-1
QDmargin = 4.5

loadpreviousrmatrix = False

if (atomicnumber == 55):
    atomStr = "Cs"
    outputRelPath = ""
    Ry = 109736.86224
    rbatomicweight = 132.9054
elif (atomicnumber == 37):
    atomStr = "Rb"
    outputRelPath = ""
    Ry = 109736.605
    rbatomicweight = 85.4678
    

    
print statelabel(nState,lState)

print "Start, End, and Step of DC Field (V/cm): {}, {}, {}".format(eminvcm, emaxvcm, estepvcm)

#------------SECTION 2-----------#


#os.chdir(csstarkDirectory)
radialMatrixElementsFilename = "rMatrixElementsMRBS.dat"


#-----------SECTION 3------------#
eunit = 5.14221e11
econversionfactor = 1/(.01 * eunit)
emin, emax, estep = eminvcm*econversionfactor, emaxvcm*econversionfactor, estepvcm*econversionfactor
numesteps = (emax-emin)/estep

#------------SECTION 4-----------#

    
mystates, mystatesFS, mystatesFSmj = getMyStates(nState,lState,maxL,QDmargin)

def myn(index):
    return mystatesFS[index,0]
    
def myl(index):
    return mystatesFS[index,1]
    
def myj(index):
    return mystatesFS[index,2]
    
def mymj(index):
    return mystatesFSmj[index,3]
    
np.savetxt("mystatesFS.dat",mystatesFS,fmt=['%5i']*3)
np.savetxt("mystatesFSmj.dat",mystatesFSmj,fmt=['%5i']*4)
    
#----------------<psi'|r|psi> Section---------------#

csRadialMatrixElement = np.zeros((mystates.shape[0],mystates.shape[0]))
if (not loadpreviousrmatrix):
    print "Calculating <psi'|r|psi> matrix elements..."
    for i, state1 in enumerate(mystates):
        for j, state2 in enumerate(mystates):
            if (state2[1] != state1[1]+1 and state2[1] != state1[1]-1):
                csRadialMatrixElement[i,j] = 0
            else:
                #print "   ...for n={}, l={}, np={}, lp={}".format(state1[0],state1[1],state2[0],state2[1])
                #print "{} {} {} {}".format(state1[0],state1[1],state2[0],state2[1])
                csRadialMatrixElement[i,j] = rme(state1[0],state1[1],state2[0],state2[1])
                #print "      result: {}".format(csRadialMatrixElement[i,j])
                
                
    np.savetxt(radialMatrixElementsFilename, csRadialMatrixElement)
    
csRadialMatrixElement = np.loadtxt(radialMatrixElementsFilename)

            
#-----------------Zero-field Hamiltonian----------#

def H0(mj):
    diagonal = np.zeros(len(mystatesFS))
    for i in range(0,len(diagonal)):
        if (abs(mj+.5)<=myj(i)+.5):
            diagonal[i] = zeroFieldEnergy(myn(i),myl(i),myj(i),atomicnumber)
    return np.diag(diagonal)

#-----------------Stark Hamiltonian---------------#

csRadialMatrixElementsFS = {}
for i, state1 in enumerate(mystates):
    for k, state2 in enumerate(mystates):
        csRadialMatrixElementsFS[twostatelabel(state1[0],state1[1],state2[0],state2[1])] = csRadialMatrixElement[i,k]
        


    
'''print "Clebsch-Gordan coeff test for: (3/2  1  5/2)"
print "                               (3/2  0  3/2)\n"
print "Using Josh's code: {}".format(clebsch(3/2,1,5/2,3/2,0,3/2))
print "Using QuTip:       {}".format(qut.clebsch(3/2,1,5/2,3/2,0,3/2))
    
quit()'''

def starkMatrixElement(i,k,mjmp5):
    n = mystatesFS[i,0]
    l = mystatesFS[i,1]
    j = mystatesFS[i,2]+.5
    np = mystatesFS[k,0]
    lp = mystatesFS[k,1]
    jp = mystatesFS[k,2]+.5
    mj = mjmp5 + 0.5
    angularMatrixElement = 0
    if (abs(mj)<=j and abs(mj)<=jp and (lp == l-1)):
        ms = -0.5
        if (sqrt((l**2-(mj-ms)**2)/((2*l+1)*(2*l-1))) != 0):
            angularMatrixElement += clebsch(l,0.5,j,mj-ms,ms,mj)*clebsch(lp,0.5,jp,mj-ms,ms,mj)*sqrt((l**2-(mj-ms)**2)/((2*l+1)*(2*l-1)))
        ms = 0.5
        if (sqrt((l**2-(mj-ms)**2)/((2*l+1)*(2*l-1))) != 0):
            angularMatrixElement += clebsch(l,0.5,j,mj-ms,ms,mj)*clebsch(lp,0.5,jp,mj-ms,ms,mj)*sqrt((l**2-(mj-ms)**2)/((2*l+1)*(2*l-1)))
    if (abs(mj)<=j and abs(mj)<=jp and (lp == l+1)):
        ms = -0.5
        if sqrt(((l+1)**2-(mj-ms)**2)/((2*l+3)*(2*l+1))) != 0:
            angularMatrixElement += clebsch(l,0.5,j,mj-ms,ms,mj)*clebsch(lp,0.5,jp,mj-ms,ms,mj)*sqrt(((l+1)**2-(mj-ms)**2)/((2*l+3)*(2*l+1)))
        ms = 0.5
        if sqrt(((l+1)**2-(mj-ms)**2)/((2*l+3)*(2*l+1))) != 0:
            angularMatrixElement += clebsch(l,0.5,j,mj-ms,ms,mj)*clebsch(lp,0.5,jp,mj-ms,ms,mj)*sqrt(((l+1)**2-(mj-ms)**2)/((2*l+3)*(2*l+1)))
    if (abs(mj)>j or abs(mj)>jp or (lp != l+1 and lp != l-1)):
        return 0
    else:
        return angularMatrixElement*csRadialMatrixElementsFS[twostatelabel(n,l,np,lp)]
        
def StarkMatrix(mj):
    mat = np.zeros((len(mystatesFS),len(mystatesFS)),dtype=np.float)
    for i in range(0,len(mystatesFS)):
        for k in range(0,len(mystatesFS)):
            if (abs(mj+.5)<= myj(i)+.5 and abs(mj+.5)<= myj(k)+.5 and (myl(i) == myl(k)+1 or myl(i) == myl(k)-1)):
                sme = starkMatrixElement(i,k,mj)
                if (not isnan(sme)):   
                    mat[i,k] = sme
    return mat

 


# TO DO...
#Rotation matrix
 
 
#rm = rotmat(0,mystatesFS)
#print "First row of rotation matrix for {}\n".format(0)
#print "State: {}".format(mystatesFS[0])
np.set_printoptions(threshold='nan')
#print np.array_str(rm[0,0:15])

 
#Calculating Hamiltonian
print "Calculating zero-AC-field Hamiltonian for mj={}...".format(0+0.5)
HstarkOverE = StarkMatrix(0)
HzeroField = H0(0)
 
ourstate = find_state([nState,lState,lState],mystatesFS)



#print HstarkOverE[find_state([82,4,3,-4],mystatesFS)]
 
#time-independent part of Hamiltonian
def Ham(e):
    return HzeroField + e*HstarkOverE


StarkEnergies = np.array([])
    
for i, e in enumerate(np.arange(emin,emax+estep,estep)):   
        
    zeromat = np.zeros((len(mystatesFS),len(mystatesFS)))


    print "Calculating Stark matrix for e={} V/cm".format(e/econversionfactor)
    print "Stark dimensions: {}x{}, {} elements.".format(len(mystatesFS),len(mystatesFS),pow(len(mystatesFS),2))
    start = time.time()
    fm = Ham(e).astype(np.float64)
    finish = time.time()
    print "Finished calculating matrix. Time elapsed: {} minutes\n".format((finish-start)/60)
    
    print "Diagonalizing Floquet matrix for e={} V/cm".format(e/econversionfactor)   
    start = time.time()    
    if (exportVecs and i==vecnum):
        val, vecs = sp.linalg.eigh(fm, overwrite_a=True, eigvals_only=False)
        lenvecs = len(vecs[0])
        format = ['%1.10e'] * lenvecs
        if (cluster != 0):
            np.savetxt("EVec_{}_DCfield{}.{}.dat".format(statelabel(nState,lState),e,cluster),vecs, fmt=format)
        else:
            np.savetxt("EVec_{}_DCfield{}.dat".format(statelabel(nState,lState),e),vecs, fmt=format)
    else:
        val = sp.linalg.eigh(fm, overwrite_a=True, eigvals_only=True)
    finish = time.time()
    print "Finished diagonalizing matrix. Time elapsed: {} minutes\n".format((finish-start)/60)
    del fm
    values = np.append(e/econversionfactor,val)
    lenvalues = len(values)
    if (len(StarkEnergies)==0):
        StarkEnergies = np.reshape(values,(1,lenvalues))
    else:
        StarkEnergies = np.append(StarkEnergies,np.reshape(values,(1,lenvalues)),axis=0)
    
print "Outputting eigenvalues..."
format = ['%1.10e'] * lenvalues   
if (cluster != 0):
    np.savetxt("Stark_{}.{}.dat".format(statelabel(nState,lState),cluster),StarkEnergies, fmt=format)
else:
    np.savetxt("Stark_{}.dat".format(statelabel(nState,lState)),StarkEnergies, fmt=format)

print "Generating graphs"
x = StarkEnergies[:,0]
for i in range(1,lenvalues,1):
    plt.plot(x,StarkEnergies[:,i])

plt.show()
plt.savefig("starkenergies.png")

print "Done at {}.".format(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

    
    
