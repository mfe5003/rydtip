
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

# returns list of states passing the n-level, energy-level, and C3 coupling strength cut at r=R0 (C3/R0**3)
def filter_molecular_states(atom, state1, state2, mol_angmon, E_range, F_cut, R0, n1_range, n2_range):
  # save states as tuples first, then convert into State c2asses
  # since I doubt that the set stuff will work correctly with classes
  newStates = set() 
  # sort the energy range to standardize
  e_low = min(abs(E_range[0]),abs(E_range[1]))
  e_high = max(abs(E_range[0]),abs(E_range[1]))
  for ang_states in mol_angmon:
    #print ang_states
    p = state1.mj - ang_states[0][-1]
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
      # now check all permutations of the n-levels in the range
      for n1b in range(n1_range[0],n1_range[1]):
        for n2b in range(n2_range[0],n2_range[1]):
          state1b = State(n1b,l1b,j1b,mj1b)
          state2b = State(n2b,l2b,j2b,mj2b)
          # check the energy first since its easier
          E = abs(consts.c*(TermEnergy(atom, state1b)[0] + TermEnergy(atom, state2b)[0]))
          if (E>e_low) and (E<e_high):
            c3 = atom.c3(state1,state2,state1b,state2b)
            if c3 != 0:
              # then check the coupling strength
              c3=c3[0] # GHz/um**3
              F = c3*(R0**3)
              if F > F_cut:
                #print(state1b,state2b,F)
                newStates.add( ((n1b,l1b,j1b,mj1b),(n2b,l2b,j2b,mj2b)) )
  return newStates


# In[8]:

if __name__ == "__main__":
  sI=State(97,2,2.5,2.5)
  E0 = consts.c*(TermEnergy(Rb87,sI)[0]+TermEnergy(Rb87,sI)[0]) # Hz
  e_range = (E0-1e9,E0+1e9)
  print(filter_molecular_states(Rb87, sI, sI, ang_mom_list, e_range, 10, 3, (50,150),(50,150)))


# In[ ]:



