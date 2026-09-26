import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import linalg
from dataclasses import dataclass

@dataclass
class Ritz_T:
    phis: list #list of shape functions
    GJ: callable
    rhoIp: callable
    length: float


    def __post_init__(self):
        self.n = len(self.phis)
        self.solvephiprimes()
        self.K = np.zeros((self.n, self.n))
        self.M = np.zeros((self.n, self.n))
        self.compute_matrices()


    def compute_matrices(self):
        for i in range(self.n):
            for j in range(self.n):
                self.K[i,j], _ = quad(lambda x: self.phiprimes[i](x)*self.phiprimes[j](x)*self.GJ(x),0,self.length)
                self.M[i,j], _ = quad(lambda x: self.phis[i](x)*self.phis[j](x)*self.rhoIp(x),0,self.length)

    def solvephiprimes(self):

        self.phiprimes = []
        for phi in self.phis:
            h=1e-6
            self.phiprimes.append(lambda x, phi=phi: (phi(x+h) - phi(x-h)) / (2*h))


    def eigs(self):

        vals, _ = linalg.eigh(self.K, self.M)

        self.omegas = np.zeros(len(vals))

        for i in range(len(vals)):

            self.omegas[i] = np.sqrt(vals[i])     


# problem 1

length = 8.0 #m

GJ = 45e6 #N*m^2

rhoIp = 7.47 #kg*m

Torsionrod = Ritz_T([lambda x: x*(x-length),lambda x: (x**2)*(x-length)],lambda x: GJ, lambda x: rhoIp, length)

Torsionrod.eigs()

print("Eigenvalues (rad/s):", Torsionrod.omegas)
print("Eigenvalues (Hz):", Torsionrod.omegas/(2*np.pi))

def GJ_var(x):

    return GJ*(4/3 - (2*x)/(3*length))**4


def rhoIp_var(x):

    return rhoIp*(4/3 - (2*x)/(3*length))**4

Torsionrod_p2 = Ritz_T([lambda x: x*(x-length)],GJ_var,rhoIp_var,length)

Torsionrod_p2.eigs()

print("Eigenvalues (rad/s):", Torsionrod_p2.omegas)
print("Eigenvalues (Hz):", Torsionrod_p2.omegas/(2*np.pi))