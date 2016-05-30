
# coding: utf-8

# In[1]:

from scipy.constants import physical_constants as pc
import scipy.constants as consts
import numpy as np
import os.path
import sqlite3
from sympy import *
from sympy.physics.wigner import clebsch_gordan, wigner_6j


# ## Common functions

# In[2]:

me = pc['electron mass']
c = pc['speed of light in vacuum']
Ry = pc['Rydberg constant']
a0 = pc['Bohr radius']
e0 = consts.epsilon_0 # this doesnt seem to be in the pc dictionary
e = consts.e
hbar = consts.hbar


# In[3]:

default_rmedb_path = os.path.join('data','RMEdbs')


# In[4]:

class Atom:
  def __init__(self, Name, Z, Inuc):
    self.Name = Name
    self.Z = Z
    self.Inuc = Inuc
    self.Configuration = -1
    self.NGround = -1
    self.FD2 = -1
    self.FD1 = -1
    self.GHz_um3_factor = 1e9*(1/(4*np.pi*e0))*e**2
    self.GHz_um3_factor *= a0[0]**2/(2*np.pi*hbar)
    try:
      self.registerRMEdb()
    except IOError:
      pass
  

  def registerRMEdb(self, db_file=None, db_path=default_rmedb_path ):
    if db_file is None:
      db_file = self.Name + '.sqlite3'
    db_file = os.path.join(db_path, db_file)
    if os.path.isfile(db_file):
      self.rme_db = db_file
    else:
      self.rme_db = None
      print("No database file found at {}".format(db_file))
      raise IOError
      
  def RMEs(self, state, n_range, l2, j2):
    if self.rme_db is None:
      raise IOError
      
    # assumes l2 >= l1, if not swap
    if int(l2) < state.l:
      if state.j == float(j2):
        j_index = 1
      elif float(j2) > int(l2):
        j_index = 2
      else:
        j_index = 0
      l_str = int(l2)*10 + state.l

    else: # otherwise do the normal order
      if state.j == float(j2):
        j_index = 1
      elif state.j > state.l:
        j_index = 2
      else:
        j_index = 0
      l_str = int(l2) + state.l*10

    js = ["jmjpm", "jpjpm", "jpjpp"][j_index]
    subs = (state.n, n_range[0], n_range[1])
    query = """SELECT n, {} FROM L{} WHERE np=? AND n>=? AND n<=?""".format(js, l_str)

    conn = sqlite3.connect(self.rme_db)
    c = conn.cursor()
    return c.execute(query, subs)
      
  def RME(self, state1, state2):
    if self.rme_db is None:
      raise IOError

    # assumes l2 >= l1, if not swap
    if state2.l < state1.l:
      n1, l1, j1, mj1 = state2.state_tuple()
      n2, l2, j2, mj2 = state1.state_tuple()
    else: # otherwise do the normal order
      n1, l1, j1, mj1 = state1.state_tuple()
      n2, l2, j2, mj2 = state2.state_tuple()

    l_str = l2 + l1*10 # matching the database table name L01: S->P
    # kind of a dumb check but whatever
    if (l2 > 9) or (l1 > 9):
      raise KeyError
      
    # pick the j levels jm = j-0.5, jp = j+0.5, j'm = j'-0.5, j'p=j'+0.5
    # jmj'p is non-physical for ED transition
    if j1 == j2:
      j_index = 1
    elif j1 > l1:
      j_index = 2
    else:
      j_index = 0

    js = ["jmjpm", "jpjpm", "jpjpp"][j_index]
    subs = (n1, n2)
    query = """SELECT {} FROM L{} WHERE n=? AND np=? LIMIT 1""".format(js, l_str)

    conn = sqlite3.connect(self.rme_db)
    c = conn.cursor()
    c.execute(query, subs)
    return [c.fetchone()[0], 'a0?', None]
  
  # angular component of matrix element
  # p is -1, 0, 1
  # We use a symbolic package to deal with the wigner symbols to avoid numerical errors
  def AME(self, state1, state2, p):
    l1, j1, m1 = state1.angular_momentum()
    l2, j2, m2 = state2.angular_momentum()
    res =(-1)**int(j1+l2-0.5)*sqrt((2*j1+1)*(2*l1+1))
    res *= clebsch_gordan(j1, 1, j2, m1, p, m2)
    res *= clebsch_gordan(l1, 1, l2, 0, 0, 0)
    res *= wigner_6j(l1,0.5,j1,j2,1,l2)
    return res
  
  # returns the c3 coefficient for the two states
  def c3(self, stateI1, stateI2, stateF1, stateF2):
    # electric dipole transitions
    if(abs(stateI1.l-stateF1.l) != 1):
      return 0
    if(abs(stateI2.l-stateF2.l) != 1):
      return 0

    p = stateF1.mj - stateI1.mj # -1,0,+1
    if abs(p)>1:
      return 0
    if stateI2.mj - stateF2.mj != p: # dmj = 0
      return 0

    a = self.AME(stateI1, stateF1, p)*self.RME(stateI1, stateF1)[0]
    b = self.AME(stateI2, stateF2, -p)*self.RME(stateI2, stateF2)[0]
    c = clebsch_gordan(1,1,2,p,-p,0)
    return [N(-self.GHz_um3_factor*sqrt(6)*c*a*b), 'GHz/um**3', None]


# In[5]:

class State:
  def __init__(self, n, l, j=None, mj=None):
    self.n = int(n)
    self.l = int(l)
    if j is None:
      self.j = j
    else:
        self.j = float(j)
    if mj is None:
      self.mj = mj
    else:
      self.mj = float(mj)
    self.make_configuration()
    
  def make_configuration(self):
    # l
    if self.l <= 2:
      self.l_label = ['S','P','D'][int(self.l)]
    else:
      self.l_label = chr(ord('F')+(int(l)-3))
    # j    
    if self.j is None:
      self.j_label = ''
    else:
      if self.j - int(self.j) < 0.5:# is j an int or half int
        self.j_label = str(int(self.j))
      else:
        self.j_label = '/'.join([str(int(2*self.j)),'2'])
        #print(self.j_label)
    # mj
    if self.mj is None:
      self.mj_label = ''
    elif abs(self.mj - int(self.mj)) < 0.5:
      self.mj_label = str(int(self.mj))
    else:
      self.mj_label = '/'.join([str(int(2*self.mj)),'2'])

  def __repr__(self):
    conf_str = str(self.n)+self.l_label
    if self.j_label != '':
      conf_str += '_'+self.j_label
      if self.mj_label != '':
        conf_str += ', mj='+self.mj_label
    return conf_str

  def state_tuple(self):
    return (self.n, self.l, self.j, self.mj)
    
  def angular_momentum(self):
    return self.state_tuple()[1:]


# In[6]:

# adds error terms in quadrature
def error_adder(*errTerms):
    total = 0
    for e in list(errTerms):
        if e is None:
            return None
        total += e**2
    return np.sqrt(total)


# In[7]:

def QD(atom, state):# TODO enter low-lying level explicitly
    n, l, j, mj = state.state_tuple()
    js, ls, ns = (str(j), str(l), str(n))
    
    qdterms = [atom.QD0, atom.QD2, atom.QD4]
    if not (j is None): # fine structure
        # first look for explicitly defined values
        try:
            return atom.QD[ns][ls][js]
        except:
            pass
        # if not a pre-defined value then calculate the value
        try:
            qds = [x[l][js][0] for x in qdterms]
        except IndexError:
            # defect is 0 for unlisted values
            return [0 , '', None]
        except KeyError:
            # possible actual error
            raise KeyError
    else: # nl basis
        try:
            js1 = str(l-0.5)
            js2 = str(l+0.5)
            if l==0:
                qds = [x[l]['0.5'][0] for x in qdterms]
            else:
                qds = [ (l*x[l][js1][0] + (l+1)*x[l][js2][0])/(2*l+1) for x in qdterms]
        except IndexError:
            # defect is 0 for unlisted values
            return [0 , '', None]
        except KeyError:
            # possible actual error
            raise KeyError
    qd = 0
    for i, x in enumerate(qds):
        qd += x/((n-qds[0])**(2*i))
    return [qd, '', None]    


# In[8]:

# returns the ionization energy for the input atom in the format
# [ value, unit, uncertainty ], similar to scipy.constants
def TermEnergy(atom, state):
    try:
        n, l, j, mj = state.state_tuple()
        js, ls, ns = (str(j), str(l), str(n))
        term = atom.TermEnergy[ns][ls][js]
        #print("predefined term energy: {}".format(term))
    except KeyError:
        # possible actual error
        if abs(float(j)-float(l)) != 0.5:
            print(n,j,l)
            raise KeyError
        # if j level is valid then perform calculation
        qd = QD(atom, state)
        try:
            uncert = error_adder(
                atom.Rydberg[2]/((int(n) - qd[0])**2),
                2*atom.Rydberg[0]*qd[2]/((int(n) - qd[0])**3)
            )
        except TypeError:
            uncert = None
        term = [-atom.Rydberg[0]/((int(n) - qd[0])**2), atom.Rydberg[1], uncert]
        #print("procedural term energy: {}".format(term))
    return term


# In[9]:

def q_adder(q1, q2):
    if q1[1] != q2[1]:
        raise UnitError
    if (q1[2] is None) or (q2[2] is None):
        uncert = None
    else:
        uncert = q1[2]+q2[2]
    return [q1[0]+q2[0], q1[1], uncert]


# In[10]:

def TransitionFrequency(atom, state1, state2):
    qd1 = QD(atom, state1)
    qd2 = QD(atom, state2)
    
    return [c[0]*atom.Rydberg[0]*((state1.n-qd1[0])**(-2)-(state2.n-qd2[0])**(-2)), 'Hz', None]


# # RB87

# ## Genernal Info

# In[11]:

Rb87=Atom('Rb87',37,1.5)
Rb87.Configuration = '[Kr]5s1'
Rb87.NGround = 5

Rb87.mass = [1.443160648e-25, 'kg', 72e-34] # Steck (2015) [4]

# simple error propagation
rmeuncert = np.sqrt((Rb87.mass[0]**4)*(me[2]**2)+(me[0]**4)*(Rb87.mass[2]**2))/((me[0]+Rb87.mass[0])**2)
Rb87.reduced_electron_mass = [me[0]/(1.0 + me[0]/Rb87.mass[0]), 'kg', rmeuncert]
Rb87.Rydberg = [Ry[0]*(Rb87.reduced_electron_mass[0]/me[0]), '1/m', None]

Rb87.TermEnergyGround = [-3369080.48, '1/m', 0.02] # ground state Hall http://dx.doi.org/10.1364/OL.3.000141


# In[12]:

Rb87.FD1 = [3.7710746322085408911e14, 'Hz', None] # Maric et al (2008) http://dx.doi.org/10.1103/PhysRevA.77.032502
Rb87.FD2 = [3.8423048457422908911e14, 'Hz', None] # Marian et al (2004) http://dx.doi.org/10.1126/science.1105660
Rb87.LambdaD1 = [c[0]/Rb87.FD1[0], 'm', None]
Rb87.LambdaD2 = [c[0]/Rb87.FD2[0], 'm', None]
Rb87.KD1 = [2*np.pi*Rb87.FD1[0]/c[0], 'rad/m', None]
Rb87.KD2 = [2*np.pi*Rb87.FD2[0]/c[0], 'rad/m', None]
Rb87.Ahfs = {
        '5S1/2': [3.417341305452145e9, 'Hz', 45e-6] # Steck (2015) from [29]
        ,'5P1/2': [408.3295e6, 'Hz', None] # Maric et al (2008) http://dx.doi.org/10.1103/PhysRevA.77.032502
        ,'5P3/2': [84.7185e6, 'Hz', 2e3] # Steck (2015) from [9]
    }
Rb87.Bhfs = {
        '5P3/2': [12.4965e6, 'Hz', 3.7e3] # Steck (2015) from [9]
    }
Rb87.Linewidth = {
        '5P': [2*np.pi*6.065e6, '1/s', None] # Mark?
        ,'5P1/2': [2*np.pi*5.7500e6, '1/s', 35e3] # Steck (2015) [18,19,21]
        ,'5P3/2': [2*np.pi*6.0666e6, '1/s', 11e3] # Steck (2015) ?
    }


# In[13]:

Rb87.IsatD2 = {
        'cycling' : [16.6933, 'W/m^2', 0.0035 ] # Steck (2015) ?
        ,'isotropic' : [35.7713, 'W/m^2', 0.0074 ] # Steck (2015) ?
        ,'pi' : [ 25.0399, 'W/m^2', 0.0052 ] # Steck (2015) ?, 
    }
Rb87.IsatD1 = {
        'pi' : [44.876, 'W/m^2', 0.031 ] # Steck (2015) ?
    }


# ## Select Experimental Spectral Lines

# In[14]:

Rb87.TermEnergy = { # explicit low lying level term energys
    '4' : { # n=4
        '2' : { #L=2 D
            '1.5' : q_adder(Rb87.TermEnergyGround, [1935564.9, '1/m', None])
            ,'2.5' : q_adder(Rb87.TermEnergyGround, [ 1935520.3, '1/m', None])
        }
        ,'3' : { #L=3 F
            '2.5' : q_adder(Rb87.TermEnergyGround, [ 2679209.2, '1/m', None])
            ,'3.5' : q_adder(Rb87.TermEnergyGround, [ 2679211.8, '1/m', None])
        }
    }
    ,'5' : { # n=5
        '0' : { #L=0 S
            '0.5' : Rb87.TermEnergyGround
        }
        ,'1' : { #L=1 P
            #'0.5' : q_adder(Rb87.TermEnergyGround, [ 1257895.098147, '1/m', None])
            #,'1.5' : q_adder(Rb87.TermEnergyGround, [ 1281654.938993, '1/m', None])
            '0.5' : q_adder(Rb87.TermEnergyGround, [ Rb87.FD1[0]/c[0], '1/m', None])
            ,'1.5' : q_adder(Rb87.TermEnergyGround, [ Rb87.FD2[0]/c[0], '1/m', None])
        }
        ,'2' : { #L=2 D
            '1.5' : q_adder(Rb87.TermEnergyGround, [ 2570053.6, '1/m', None])
            ,'2.5' : q_adder(Rb87.TermEnergyGround, [ 2570349.8, '1/m', None])
        }
    }
    ,'6' : { # n=6
        '0' : { #L=0 S
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 2013251.0, '1/m', None])
        }
        ,'1' : { #L=1 P
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 2371508.1, '1/m', None])
            ,'1.5' : q_adder(Rb87.TermEnergyGround, [ 2379259.1, '1/m', None])
        }
        ,'2' : { #L=2 D
            '1.5' : q_adder(Rb87.TermEnergyGround, [ 2868712.7, '1/m', None])
            ,'2.5' : q_adder(Rb87.TermEnergyGround, [ 2868939.0, '1/m', None])
        }
    }
    ,'7' : { # n=7
        '0' : { #L=0 S
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 2631143.7, '1/m', None])
        }
        ,'1' : { #L=1 P
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 2783502., '1/m', None])
            ,'1.5' : q_adder(Rb87.TermEnergyGround, [ 2787011., '1/m', None])
        }
        ,'2' : { #L=2 D
            '1.5' : q_adder(Rb87.TermEnergyGround, [ 2868712.7, '1/m', None])
            ,'2.5' : q_adder(Rb87.TermEnergyGround, [ 2868939.0, '1/m', None])
        }
    }
    ,'8' : { # n=8
        '0' : { #L=0 S
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 2904681.6, '1/m', None])
        }
        ,'1' : { #L=1 P
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 2983494., '1/m', None])
            ,'1.5' : q_adder(Rb87.TermEnergyGround, [ 2985379., '1/m', None])
        }
    }
    ,'9' : { # n=9
        '1' : { #L=1 P
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 3095894., '1/m', None])
            ,'1.5' : q_adder(Rb87.TermEnergyGround, [ 3097022., '1/m', None])
        }
    }
    ,'10' : { # n=10
        '1' : { #L=1 P
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 3165388., '1/m', None])
            ,'1.5' : q_adder(Rb87.TermEnergyGround, [ 3166119., '1/m', None])
        }
    }
    ,'11' : { # n=11
        '1' : { #L=1 P
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 3211358., '1/m', None])
            ,'1.5' : q_adder(Rb87.TermEnergyGround, [ 3211855., '1/m', None])
        }
    }
    ,'12' : { # n=11
        '1' : { #L=1 P
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 3243350., '1/m', None])
            ,'1.5' : q_adder(Rb87.TermEnergyGround, [ 3243704., '1/m', None])
        }
    }
}


# ## Quantum Defects

# In[15]:

Rb87.QD0 = [ # 0th order qd terms
    { # L = 0, S
        '0.5' : [3.1311804, '', None]
    }
    ,{ # L = 1, P
        '0.5' : [2.6548849, '', None]
        ,'1.5': [2.6416737, '', None]
    }
    ,{ # L = 2, D
        '1.5' : [1.34809171, '', None]
        ,'2.5': [1.34646572, '', None]
    }
    ,{ # L = 3, F
        '2.5' : [0.0165192, '', None]
        ,'3.5': [0.0165437, '', None]
    }
]
Rb87.QD2 = [ # 2nd order qd terms
    { # L = 0, S
        '0.5' : [0.1784, '', None]
    }
    ,{ # L = 1, P
        '0.5' : [0.2900, '', None]
        ,'1.5': [0.2950, '', None]
    }
    ,{ # L = 2, D
        '1.5' : [-0.60286, '', None]
        ,'2.5': [-0.59600, '', None]
    }
    ,{ # L = 3, F
        '2.5' : [-0.085, '', None]
        ,'3.5': [-0.086, '', None]
    }
]
Rb87.QD4 = [ # 4th order qd terms
    { # L = 0, S
        '0.5' : [-1.8, '', None]
    }
    ,{ # L = 1, P
        '0.5' : [-7.904, '', None]
        ,'1.5': [-0.97495, '', None]
    }
    ,{ # L = 2, D
        '1.5' : [-1.50517, '', None]
        ,'2.5': [-1.50517, '', None]
    }
    ,{ # L = 3, F
        '2.5' : [-0.36005, '', None]
        ,'3.5': [-0.36005, '', None]
    }
]


# ## Generating Quantum Defects from Explicit Spectral Lines

# In[16]:

# calculate defects for low-lying levels from the spectroscopy data
Rb87.QD = {}
for n, nd in Rb87.TermEnergy.iteritems():
    Rb87.QD[n] = {}
    for l, ld in nd.iteritems():
        Rb87.QD[n][l]={}
        for j, jd in ld.iteritems():
            term = TermEnergy(Rb87, State(n, l, j))
            try:
                uncert = error_adder( Rb87.Rydberg[2]/(2*np.sqrt(-Rb87.Rydberg[0]*term[0])), term[2]*np.sqrt(-Rb87.Rydberg[0]/(4*(term[0]**3))) )
            except TypeError:
                uncert = None
            Rb87.QD[n][l][j] = [int(n) - np.sqrt(-Rb87.Rydberg[0]/term[0]), '', uncert]


# ## Verification of Defects with respect to Mark's old code

# In[17]:

QD(Rb87,State(9,1,1.5))


# In[18]:

#sI = State(5,0,0.5)
test_cases = [
    ( QD(Rb87, State(20,0,0.5)), 3.13178510955 ),
    ( QD(Rb87, State(9,1,1.5)), 2.64897056637 ),
    ( QD(Rb87, State(9,2,2.5)), 1.33629100633 ),
    ( QD(Rb87, State(9,3,3.5)), 0.0154780575013 ),
    ( QD(Rb87, State(9,4,4.5)), 0 ),
    ( QD(Rb87, State(20,0)), 3.13178510955),
    ( QD(Rb87, State(30,1)), 2.64646359904),
    ( QD(Rb87, State(9,2)), 1.33645400157),
    ( QD(Rb87, State(9,3)), 0.0154175880834),
    ( QD(Rb87, State(9,4)), 0),
    ( QD(Rb87, State(5,0,0.5)), 3.195237315299605),
    ( QD(Rb87, State(5,1,1.5)), 2.70717821684838),
    ( QD(Rb87, State(5,2,2.5)), 1.2934),
    ( QD(Rb87, State(6,0,0.5)), 3.15506),
    ( QD(Rb87, State(6,1,1.5)), 2.67036),
]

for qd_new, qd_old in test_cases:
    if __name__ == "__main__":
        print("{:.6f} ({:.6f}): {:.6f}".format(qd_new[0], qd_old, qd_new[0]- qd_old))


# In[19]:

s1 = State(5,0,0.5)
s2 = State(5,1,1.5)
TransitionFrequency(Rb87, s1, s2)


# In[20]:

s1 = State(5,1,1.5)
s2 = State(97,2,2.5)
if __name__ == "__main__":
    print(TransitionFrequency(Rb87, s1, s2))
c[0]/TransitionFrequency(Rb87, s1, s2)[0]


# ## Verification of Radial Matrix Elements

# In[21]:

try:
  Rb87.registerRMEdb(None, os.path.join('..',default_rmedb_path)) # just use the default
except IOError:
  try:
    Rb87.registerRMEdb() # just use the default
  except IOError:
    print('this is dumb')

test_cases = [
  ( Rb87.RME(State(12, 2, 1.5), State(11, 1, 1.5)), -2.51450167632 ),
  ( Rb87.RME(State(11, 1, 1.5), State(12, 2, 1.5)), -2.51450167632 ),
  ( Rb87.RME(State(11, 1, 0.5), State(12, 2, 1.5)), -2.03519312761 ),
]

for rme_new, rme_old in test_cases:
    if __name__ == "__main__":
        print("{:.6f} ({:.6f}): {:.6f}".format(rme_new[0], rme_old, rme_new[0]- rme_old))


# ## Verification of Angular Matrix Elements

# In[22]:

N(Rb87.AME(State(5,1,1.5,0.5),State(5,2,2.5,1.5),1))


# In[23]:

Rb87.AME(State(5,1,1.5,0.5),State(5,2,2.5,1.5),1)


# ## Verification of C_3 terms

# In[24]:

s0=State(97,2,2.5,2.5)
sf1=State(99,1,1.5,1.5)
sf2=State(95,3,3.5,3.5)
 # check against older values

sf1a=State(98,1,1.5,1.5)
sf2a=State(96,3,3.5,3.5)

test_cases = [
  ( Rb87.c3(s0,s0,sf1,sf2), 21.7965 ),
  ( Rb87.c3(s0,s0,sf1a,sf2a), 61.1639 ),
]

for c3_new, c3_old in test_cases:
    if __name__ == "__main__":
        print("{:.6f} ({:.6f}): {:.6f}".format(c3_new[0], c3_old, c3_new[0]- c3_old))

