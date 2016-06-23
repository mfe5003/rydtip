from quantumdefects import *
try:
    from sympy.mpmath import *
except:
    from mpmath import *

def DAngerJ(nu, z):
    return 0.5 * (angerj(nu-1,z)-angerj(nu+1,z))
    
def nuc(nu1,nu2):
    return pow(2*pow(nu1*nu2,2)/(nu1+nu2),1.0/3.0)
    
def ex(l1,l2,nu1,nu2):
    return sqrt(1-pow((l1+l2+1)/(2*nuc(nu1,nu2)),2))
    
def dr(exx,s,l2,l1):
    if abs(s)<1e-10:
        s=s+1e-10
    return (1-exx)*sinc(pi*s) + (1/s)*(DAngerJ(-s,exx*s)+np.sign(l2-l1)*sqrt(pow(exx,-2)-1)*(angerj(-s,exx*s)-sinc(pi*s)))

def rme(n1,l1,n2,l2):
    d1 = deltaQD(n1,l1,l1)*((l1+1.)/(2*l1+1))+deltaQD(n1,l1,l1-1)*((l1*1.0)/(2*l1+1))
    d2 = deltaQD(n2,l2,l2)*((l2+1.)/(2*l2+1))+deltaQD(n2,l2,l2-1)*((l2*1.0)/(2*l2+1))
    #print "QD for n1={}, l1={}: {}".format(n1,l1,d1)
    #print "QD for n2={}, l2={}: {}".format(n2,l2,d2)
    return pow(-1,n1-n2)*pow(nuc(n1-d1,n2-d2),5)*pow((n1-d1)*(n2-d2),-3./2.)*dr(ex(l1,l2,n1-d1,n2-d2),n2-d2-(n1-d1),l2,l1)