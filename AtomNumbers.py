
# coding: utf-8

# In[1]:

from scipy.constants import physical_constants as pc
import numpy as np


# ## Common functions

# In[2]:

me = pc['electron mass']
c = pc['speed of light in vacuum']
Ry = pc['Rydberg constant']


# In[3]:

class Atom:
    def __init__(self, Name, Z, Inuc):
        self.Name = Name
        self.Z = Z
        self.Inuc = Inuc
        self.Configuration = -1
        self.NGround = -1
        self.FD2 = -1
        self.FD1 = -1


# In[4]:

# adds error terms in quadrature
def error_adder(*errTerms):
    total = 0
    for e in list(errTerms):
        if e is None:
            return None
        total += e**2
    return np.sqrt(total)


# In[74]:

def QD(atom, n, l, j=-1):# TODO enter low-lying level explicitly
    js=str(j)
    ls=str(l)
    ns=str(n)
    
    qdterms = [atom.QD0, atom.QD2, atom.QD4]
    if j>0: # fine structure
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


# In[75]:

# returns the ionization energy for the input atom in the format
# [ value, unit, uncertainty ], similar to scipy.constants
def TermEnergy(atom, n, l, j):
    try:
        js = str(j)
        ls = str(l)
        ns = str(n)
        # if the 
        term = atom.TermEnergy[ns][ls][js]
        #print("predefined term energy: {}".format(term))
    except KeyError:
        # possible actual error
        if abs(float(j)-float(l)) != 0.5:
            print(n,j,l)
            raise KeyError
        # if j level is valid then perform calculation
        qd = QD(atom, int(n), l, j)
        try:
            uncert = error_adder(
                atom.Rydberg[2]/((int(n) - qd[0])**2),
                2*atom.Rydberg[0]*qd[2]/((int(n) - qd[0])**3)
            )
        except TypeError:
            uncert = None
        term = [-atom.Rydberg[0]/((int(n) - qd[0])**2), atom.Rydberg[1], uncert]
        print("procedural term energy: {}".format(term))
    return term


# In[76]:

def q_adder(q1, q2):
    if q1[1] != q2[1]:
        raise UnitError
    if (q1[2] is None) or (q2[2] is None):
        uncert = None
    else:
        uncert = q1[2]+q2[2]
    return [q1[0]+q2[0], q1[1], uncert]


# # RB87

# ## Genernal Info

# In[93]:

Rb87=Atom('Rb87',37,1.5)
Rb87.Configuration = '[Kr]5s1'
Rb87.NGround = 5

Rb87.mass = [1.443160648e-25, 'kg', 72e-34] # Steck (2015) [4]

# simple error propagation
rmeuncert = np.sqrt((Rb87.mass[0]**4)*(me[2]**2)+(me[0]**4)*(Rb87.mass[2]**2))/((me[0]+Rb87.mass[0])**2)
Rb87.reduced_electron_mass = [me[0]/(1.0 + me[0]/Rb87.mass[0]), 'kg', rmeuncert]
Rb87.Rydberg = [Ry[0]*(Rb87.reduced_electron_mass[0]/me[0]), '1/m', None]

Rb87.TermEnergyGround = [-3369080.48, '1/m', None] # ground state Hall http://dx.doi.org/10.1364/OL.3.000141


# In[78]:

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


# In[79]:

Rb87.IsatD2 = {
        'cycling' : [16.6933, 'W/m^2', 0.0035 ] # Steck (2015) ?
        ,'isotropic' : [35.7713, 'W/m^2', 0.0074 ] # Steck (2015) ?
        ,'pi' : [ 25.0399, 'W/m^2', 0.0052 ] # Steck (2015) ?, 
    }
Rb87.IsatD1 = {
        'pi' : [44.876, 'W/m^2', 0.031 ] # Steck (2015) ?
    }


# ## Select Experimental Spectral Lines

# In[102]:

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
            '0.5' : q_adder(Rb87.TermEnergyGround, [ 1257895.098147, '1/m', None])
            ,'1.5' : q_adder(Rb87.TermEnergyGround, [ 1281654.938993, '1/m', None])
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

# In[103]:

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

# In[104]:

# calculate defects for low-lying levels from the spectroscopy data
Rb87.QD = {}
for n, nd in Rb87.TermEnergy.iteritems():
    Rb87.QD[n] = {}
    for l, ld in nd.iteritems():
        Rb87.QD[n][l]={}
        for j, jd in ld.iteritems():
            term = TermEnergy(Rb87, n, l, j)
            try:
                uncert = error_adder( Rb87.Rydberg[2]/(2*np.sqrt(Rb87.Rydberg[0]*term[0])), term[2]*np.sqrt(Rb87.Rydberg[0]/(4*(term[0]**3))) )
            except TypeError:
                uncert = None
            Rb87.QD[n][l][j] = [int(n) - np.sqrt(-Rb87.Rydberg[0]/term[0]), '', uncert]


# ## Verification of Defects with respect to Mark's old code

# In[105]:

QD(Rb87,9,1,1.5)


# In[106]:

print(QD(Rb87,20,0,0.5)[0]-3.13178510955)
print(QD(Rb87,9,1,1.5)[0]-2.64897056637)
print(QD(Rb87,9,2,2.5)[0]-1.33629100633)
print(QD(Rb87,9,3,3.5)[0]-0.0154780575013)
print(QD(Rb87,9,4,4.5)[0]-0)


# In[107]:

print(QD(Rb87,20,0)[0]-3.13178510955)
print(QD(Rb87,30,1)[0]-2.64646359904)
print(QD(Rb87,9,2)[0]-1.33645400157)
print(QD(Rb87,9,3)[0]-0.0154175880834)
print(QD(Rb87,9,4)[0]-0)


# In[108]:

print(QD(Rb87,5,0,0.5)[0]-3.195237315299605)
print(QD(Rb87,5,1,1.5)[0]-2.70717821684838)
print(QD(Rb87,5,2,2.5)[0]-1.2934)
print(QD(Rb87,6,0,0.5)[0]-3.15506)
print(QD(Rb87,6,1,1.5)[0]-2.67036)


# In[ ]:



