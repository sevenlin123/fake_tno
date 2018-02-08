import numpy as np
from skyfield.api import load, Topos
from scipy.optimize import newton
planets = load('de423.bsp')

class plutino:
    """
    generate plutinos
    
    """
    def __init__(self):
        self.size = 1000
        self.lambda_N = 0
        self.a = self.gen_a()
        self.e = self.gen_e()
        self.i = self.gen_i()
        self.amp = self.gen_amp()
        self.phi = self.gen_phi(self.amp)
        self.M = self.gen_M()
        self.node = self.gen_node()
        self.arg = self.gen_arg(self.phi, self.M, self.node, self.lambda_N)
    
    def gen_a(self):
        return 39.45 + np.random.random(self.size) * 0.4 - 0.2
        
    def gen_e(self):
        return np.random.normal(loc=0.175, scale=0.06, size = self.size)
        
    def gen_i(self):
        return np.arcsin(np.random.rayleigh(scale = 12 * np.pi/180., size = self.size))*180/np.pi
        
    def gen_amp(self):
        return np.random.triangular(0, 75, 155, size = self.size)
    
    def gen_phi(self, amp):
        return 180 + amp * np.sin(2*np.pi*np.random.random(self.size))
        
    def gen_M(self):
        return 2*np.pi*np.random.random(self.size)
        
    def gen_node(self):
        return 2*np.pi*np.random.random(self.size)
        
    def gen_arg(self, phi, M, node, lambda_N):
        return 0.5*phi - 1.5*M - node + lambda_N
        
    def gen_H(self):
        alpha = 0.9
        h0 = 0
        h1 = 10
        h0s10 = 10**(alpha*h0)
        h1s10 = 10**(alpha*h1)
        return np.log10(np.random.random(self.size)*(h1s10-h0s10) + h0s10) / alpha
        
    def kep_to_xyz(a, e, i, arg, node, M):
        f = lambda E, M, e: E-e*np.sin(E)-M
        E0 = M
        E = newton(f, E0, args=(M, e))
        
        
def main():
    p = plutino()
    

if __name__ == '__main__':
    main()