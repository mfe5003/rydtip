from math import pi


def statelabel(n,l):
    label = str(n)
    if (l == 0):
        label = label+"S"
    elif (l == 1):
        label = label+"P"
    elif (l == 2):
        label = label+"D"
    elif (l == 3):
        label = label+"F"
    elif (l == 4):
        label = label+"G"
    elif (l == 5):
        label = label+"H"
    else:
        label = label+str(l)
    return label

def twostatelabel(n,l,np,lp):
    return statelabel(n,l)+"+"+statelabel(np,lp)

#This converts a microwave intensity in Watts per square centimeter to an electric field in atomic units.
def acfConvertToAU(field):
    return field*5.33802e-9
    
#This converts a microwave field amplitude in Volta per centimeter to an electric field in atomic units.
def acfConvertVpcmToAU(field):
    eunit = 5.14221e11
    econversionfactor = 1/(.01 * eunit)
    return field*econversionfactor

#This converts frequency in Hz to energy in atomic units.
def energyw(w):
    return w*2.41885e-17*2*pi

def find_state(state,mystatesFS):
    cond = map(lambda x: x==state, mystatesFS)
    for i, k in enumerate(cond):
        if k.all():
            return i
    return -1