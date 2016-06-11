
# coding: utf-8

# In[1]:

from AtomNumbers import State, TermEnergy, Rb87
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import scipy.constants as consts
from sympy import *
from sympy.physics.wigner import clebsch_gordan


# In[2]:

def getRelevantCouplings(couplings, forsters, r, ccut, ecut = 1):
    # get the center molecular state (energy is 0 by design)
    cindex = np.abs(forsters).argsort()[0]
    r3 = r**3
    # make the hamiltonian at r
    H=(np.divide(couplings, r3) + np.diag(forsters))
    # do the thing
    e, v = np.linalg.eig(H)
    v = np.asarray(v)
    #print(e)
    # save the data that has coupling greater than ccut
    data = []
    for i in range(len(v)):
        if (abs(v[cindex][i]) >= ccut) and (abs(e[i]) <= ecut):
            data.append([ r, np.real(e[i]), (np.real(v[cindex][i]))**2, np.abs(v[:,i]).argsort()[-5:][::-1] ])
    
    return data #[ r(um), E(r) (GHz), overlap with center state, largest overlaps from basis states ]


# In[3]:

# return a list of states that ED couple to the input states
def next_ED_order(m_states):
  newStates=set() # set class maintains uniqueness automatically
  for ms in m_states:
    m_total = ms[0][0].mj + ms[0][1].mj # total angualr momentum is conserved
    for new_ms in itertools.product(coupled_angmom_states(ms[0][0]), coupled_angmom_states(ms[0][1])):
      if (new_ms[0][-1] + new_ms[1][-1]) == m_total:
        newStates.add(new_ms)
  return newStates


# In[4]:

# return a set of angular momentum states that could ED couple to the input state
def coupled_angmom_states(state):
  all_states = []
  ll = state.l
  for l in np.arange(abs(ll-1),ll+1.1,2):
    for j in np.arange(abs(l-0.5),l+0.6):
      if abs(j-state.j) <= 1:
        for mj in np.linspace(-j,j,int(2*j+1)):
          if abs(mj-state.mj) <= 1:
            all_states.append((l,j,mj))
  return all_states


# In[5]:

coupled_angmom_states(State(97,2,2.5,2.5))


# In[6]:

ang_mom_list=list(next_ED_order([[(State(97,2,2.5,2.5),State(97,2,2.5,2.5)),5555]]))


# In[7]:

ang_mom_list


# In[19]:

# returns list of states passing the n-level, energy-level, and C3 coupling strength cut at r=R0 (C3/R0**3)
def filter_molecular_states(atom, state1, state2, mol_angmon, F_cut, R0, n1_range, n2_range):
  # save states as tuples first, then convert into State classes
  # since I doubt that the set stuff will work correctly with classes
  newStates = set()
  E0 = abs(consts.c*(TermEnergy(atom, state1)[0] + TermEnergy(atom, state2)[0])) # Hz
  for ang_states in mol_angmon:
    #print ang_states
    p = ang_states[0][-1] - state1.mj
    dl1 = state1.l - ang_states[0][0]
    dl2 = state2.l - ang_states[1][0]
    if (abs(p)<=1) and (abs(dl1)==1) and (abs(dl2)==1):   
      # rme database lookups
      l1b, j1b, mj1b = ang_states[0]
      l2b, j2b, mj2b = ang_states[1]
      # precalculate angular factors
      ang_factor = atom.AME(state1, State(0,l1b,j1b,mj1b), p)
      ang_factor *= atom.AME(state2, State(0,l2b,j2b,mj2b), -p)
      ang_factor *= clebsch_gordan(1,1,2,p,-p,0)
      # calculate threshold to add molecular state to the basis
      if ang_factor != 0:
        cut = abs(N(F_cut*R0**3/(ang_factor*atom.GHz_um3_factor*sqrt(6.0))))
        #print("cut: {}".format(cut))
        # now check all permutations of the n-levels in the range
        s1bs = [ [ (x[0],)+ang_states[0], x[1]] for x in atom.RMEs(state1,n1_range,l1b,j1b) ]
        s2bs = [ [ (x[0],)+ang_states[1], x[1]] for x in atom.RMEs(state2,n2_range,l2b,j2b) ]
        for ms in itertools.product(s1bs,s2bs):
          state1b = State(ms[0][0][0],ms[0][0][1],ms[0][0][2],ms[0][0][3])
          state2b = State(ms[1][0][0],ms[1][0][1],ms[1][0][2],ms[1][0][3])
          # check the energy first since its easier
          E = abs(consts.c*(TermEnergy(atom, state1b)[0] + TermEnergy(atom, state2b)[0])) # Hz
               
          if abs(ms[0][1]*ms[1][1]/((E-E0)*1e-9)) > cut:
            #print("E (GHz): {}".format((E-E0)*1e-9))
            newStates.add((ms[0][0],ms[1][0]))
  return newStates


# In[20]:

if __name__ == "__main__":
  sI=State(97,2,2.5,2.5)
  print(filter_molecular_states(Rb87, sI, sI, ang_mom_list, 1, 3, (80,120),(80,120)))


# In[ ]:



