
# coding: utf-8

# In[1]:

from scipy.constants import physical_constants as pc
import numpy as np


# ## Common functions

# In[2]:

me = pc['electron mass'][0]
c = pc['speed of light in vacuum'][0]
Ry = pc['Rydberg constant'][0]


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

def QD(atom, n, l, j=-1):# TODO enter low-lying level explicitly
    js=str(j)
    ls=str(l)
    ns=str(n)
    if j>0: # fine structure
        # first look for explicitly defined values
        try:
            qd = atom.QD[ns][ls][js]
            #print("explicit defect found")
            return qd
        except:
            #print("no explicit defect found")
            pass
        # if not a pre-defined value then calculate the value
        try:
            qd0 = atom.QD0[l][js]
            qd2 = atom.QD2[l][js]
            qd4 = atom.QD4[l][js]
        except IndexError:
            # defect is 0 for unlisted values
            return 0
        except KeyError:
            # possible actual error
            raise KeyError
    else:
        try:
            js1 = str(l-0.5)
            js2 = str(l+0.5)
            if l==0:
                qd0 = atom.QD0[l][js2]
                qd2 = atom.QD2[l][js2]
                qd4 = atom.QD4[l][js2]
            else:
                qd0 = (l*atom.QD0[l][js1] + (l+1)*atom.QD0[l][js2])/(2*l+1)
                qd2 = (l*atom.QD2[l][js1] + (l+1)*atom.QD2[l][js2])/(2*l+1)
                qd4 = (l*atom.QD4[l][js1] + (l+1)*atom.QD4[l][js2])/(2*l+1)
        except IndexError:
            # defect is 0 for unlisted values
            return 0
        except KeyError:
            # possible actual error
            raise KeyError
    return qd0 + qd2/((n-qd0)**2) + qd4/((n-qd0)**4)
    


# In[5]:

def TermEnergy(atom, n, l, j):
    try:
        js = str(j)
        ls = str(l)
        ns = str(n)
        # if the 
        term = atom.TermEnergy[ns][ls][js]
    except KeyError:
        # possible actual error
        if abs(float(j)-float(l)) != 0.5:
            print(n,j,l)
            raise KeyError
        # if j level is valid then perform calculation
        term = -atom.RyCGS/(int(n) - QD(atom, int(n), l, j)**2)
    return term


# # RB87

# ## Genernal Info

# In[6]:

Rb87=Atom('Rb87',37,1.5)
Rb87.Configuration = '[Kr]5s1'
Rb87.NGround = 5
Rb87.FD1 = 3.7710746322085408911e14 # Maric et al (2008) http://dx.doi.org/10.1103/PhysRevA.77.032502
Rb87.FD2 = 3.8423048457422908911e14 # Marian et al (2004) http://dx.doi.org/10.1126/science.1105660
Rb87.LambdaD1 = c/Rb87.FD1
Rb87.LambdaD2 = c/Rb87.FD2
Rb87.KD1 = 2*np.pi*Rb87.FD1/c
Rb87.KD2 = 2*np.pi*Rb87.FD2/c
Rb87.Ahfs = {
        '5S1/2': 3.417341305452145e9 # Steck (2015) from [29]
        ,'5P1/2': 408.3295e6 # Maric et al (2008) http://dx.doi.org/10.1103/PhysRevA.77.032502
        ,'5P3/2': 84.7185e6 # Steck (2015) from [9]
    }
Rb87.Bhfs = {
        '5P3/2': 12.4965e6 # Steck (2015) from [9]
    }
Rb87.Linewidth = {
        '5P': 2*np.pi*6.065e6 # Mark?
        ,'5P1/2': 2*np.pi*5.7500e6 # Steck (2015) [18,19,21]
        ,'5P3/2': 2*np.pi*6.0666e6 # Steck (2015) ?
    }
Rb87.MassSI = 1.443160648e-25 # Steck (2015) [4]
Rb87.RmeSI = me/(1.0 + me/Rb87.MassSI)
Rb87.RySI = Ry*(Rb87.RmeSI/me) # m^-1
Rb87.RyCGS = Rb87.RySI/100 # cm^-1
Rb87.IsatD2 = {
        'cycling' : 16.6933 # W/m^2 Steck (2015) ?
        ,'isotropic' : 35.7713 # W/m^2 Steck (2015) ?
        ,'pi' : 25.0399 # W/m^2 Steck (2015) ?
    }
Rb87.IsatD1 = {
        'pi' : 44.876 # W/m^2 Steck (2015) ?
    }
Rb87.TermEnergyGround = -33690.8048 # cm^-1 ground state Hall http://dx.doi.org/10.1364/OL.3.000141


# ## Select Experimental Spectral Lines

# In[7]:

Rb87.TermEnergy = { # explicit low lying level term energys
    '4' : { # n=4
        '2' : { #L=2 D
            '1.5' : Rb87.TermEnergyGround + 19355.649 # cm^-1
            ,'2.5' : Rb87.TermEnergyGround + 19355.203 # cm^-1
        }
        ,'3' : { #L=3 F
            '2.5' : Rb87.TermEnergyGround + 26792.092 # cm^-1
            ,'3.5' : Rb87.TermEnergyGround + 26792.118 # cm^-1
        }
    }
    ,'5' : { # n=5
        '0' : { #L=0 S
            '0.5' : Rb87.TermEnergyGround
        }
        ,'1' : { #L=1 P
            '0.5' : Rb87.TermEnergyGround + 12578.95098147 # cm^-1
            ,'1.5' : Rb87.TermEnergyGround + 12816.54938993 # cm^-1
        }
        ,'2' : { #L=2 D
            '1.5' : Rb87.TermEnergyGround + 25700.536 # cm^-1
            ,'2.5' : Rb87.TermEnergyGround + 25703.498 # cm^-1
        }
    }
    ,'6' : { # n=6
        '0' : { #L=0 S
            '0.5' : Rb87.TermEnergyGround + 20132.510
        }
        ,'1' : { #L=1 P
            '0.5' : Rb87.TermEnergyGround + 23715.081 # cm^-1
            ,'1.5' : Rb87.TermEnergyGround + 23792.591 # cm^-1
        }
        ,'2' : { #L=2 D
            '1.5' : Rb87.TermEnergyGround + 28687.127 # cm^-1
            ,'2.5' : Rb87.TermEnergyGround + 28689.390 # cm^-1
        }
    }
    ,'7' : { # n=7
        '0' : { #L=0 S
            '0.5' : Rb87.TermEnergyGround + 26311.437
        }
        ,'1' : { #L=1 P
            '0.5' : Rb87.TermEnergyGround + 27835.02 # cm^-1
            ,'1.5' : Rb87.TermEnergyGround + 27870.11 # cm^-1
        }
        ,'2' : { #L=2 D
            '1.5' : Rb87.TermEnergyGround + 28687.127 # cm^-1
            ,'2.5' : Rb87.TermEnergyGround + 28689.390 # cm^-1
        }
    }
    ,'8' : { # n=8
        '0' : { #L=0 S
            '0.5' : Rb87.TermEnergyGround + 29046.816
        }
        ,'1' : { #L=1 P
            '0.5' : Rb87.TermEnergyGround + 29834.94 # cm^-1
            ,'1.5' : Rb87.TermEnergyGround + 29853.79 # cm^-1
        }
    }
    ,'9' : { # n=9
        '1' : { #L=1 P
            '0.5' : Rb87.TermEnergyGround + 30958.94 # cm^-1
            ,'1.5' : Rb87.TermEnergyGround + 30970.22 # cm^-1
        }
    }
    ,'10' : { # n=10
        '1' : { #L=1 P
            '0.5' : Rb87.TermEnergyGround + 31653.88 # cm^-1
            ,'1.5' : Rb87.TermEnergyGround + 31661.19 # cm^-1
        }
    }
    ,'11' : { # n=11
        '1' : { #L=1 P
            '0.5' : Rb87.TermEnergyGround + 32113.58 # cm^-1
            ,'1.5' : Rb87.TermEnergyGround + 32118.55 # cm^-1
        }
    }
    ,'12' : { # n=11
        '1' : { #L=1 P
            '0.5' : Rb87.TermEnergyGround + 32433.50 # cm^-1
            ,'1.5' : Rb87.TermEnergyGround + 32437.04 # cm^-1
        }
    }
}


# ## Quantum Defects

# In[8]:

Rb87.QD0 = [ # 0th order qd terms
    { # L = 0, S
        '0.5' : 3.1311804
    }
    ,{ # L = 1, P
        '0.5' : 2.6548849
        ,'1.5': 2.6416737
    }
    ,{ # L = 2, D
        '1.5' : 1.34809171
        ,'2.5': 1.34646572
    }
    ,{ # L = 3, F
        '2.5' : 0.0165192
        ,'3.5': 0.0165437
    }
]
Rb87.QD2 = [ # 2nd order qd terms
    { # L = 0, S
        '0.5' : 0.1784
    }
    ,{ # L = 1, P
        '0.5' : 0.2900
        ,'1.5': 0.2950
    }
    ,{ # L = 2, D
        '1.5' : -0.60286
        ,'2.5': -0.59600
    }
    ,{ # L = 3, F
        '2.5' : -0.085
        ,'3.5': -.086
    }
]
Rb87.QD4 = [ # 4th order qd terms
    { # L = 0, S
        '0.5' : -1.8
    }
    ,{ # L = 1, P
        '0.5' : -7.904
        ,'1.5': -0.97495
    }
    ,{ # L = 2, D
        '1.5' : -1.50517
        ,'2.5': -1.50517
    }
    ,{ # L = 3, F
        '2.5' : -.36005
        ,'3.5': -.36005
    }
]


# ## Generating Quantum Defects from Explicit Spectral Lines

# In[9]:

# calculate defects for low-lying levels from the spectroscopy data
Rb87.QD = {}
for n, nd in Rb87.TermEnergy.iteritems():
    Rb87.QD[n] = {}
    for l, ld in nd.iteritems():
        Rb87.QD[n][l]={}
        for j, jd in ld.iteritems():
            Rb87.QD[n][l][j] = int(n) - np.sqrt(-Rb87.RyCGS/TermEnergy(Rb87, n, l, j))


# ## Verification of Defects with respect to Mark's old code

# In[10]:

print(QD(Rb87,20,0,0.5)-3.13178510955)
print(QD(Rb87,9,1,1.5)-2.64897056637)
print(QD(Rb87,9,2,2.5)-1.33629100633)
print(QD(Rb87,9,3,3.5)-0.0154780575013)
print(QD(Rb87,9,4,4.5)-0)


# In[11]:

print(QD(Rb87,20,0)-3.13178510955)
print(QD(Rb87,30,1)-2.64646359904)
print(QD(Rb87,9,2)-1.33645400157)
print(QD(Rb87,9,3)-0.0154175880834)
print(QD(Rb87,9,4)-0)


# In[12]:

print(QD(Rb87,5,0,0.5)-3.195237315299605)
print(QD(Rb87,5,1,1.5)-2.70717821684838)
print(QD(Rb87,5,2,2.5)-1.2934)
print(QD(Rb87,6,0,0.5)-3.15506)
print(QD(Rb87,6,1,1.5)-2.67036)

