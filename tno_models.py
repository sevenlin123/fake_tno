import numpy as np
from skyfield.api import load, Topos
from scipy.optimize import newton
planets = load('de423.bsp')


class Resonant:
    """
    The resonant class
    units: au, radius
    
    """
    def __init__(self):
        self.size = None
        self.mjd = None
        self.lambda_N = None
        
    def gen_e(self, e_c, e_sigma):
        return np.random.normal(loc=e_c, scale=e_sigma, size = self.size)
        
    def gen_i(self, i_sigma):
        return np.arcsin(np.random.rayleigh(scale = i_sigma * np.pi/180., size = self.size))
        
    def gen_amp(self, amp_c, amp_max, amp_min):
        return np.random.triangular(amp_min, amp_c, amp_max, size = self.size) * np.pi / 180.
        
    def gen_node(self):
        return 2*np.pi*np.random.random(self.size) % (2*np.pi)
        
    def gen_H(self, h0, h1):
        alpha = 0.9
        h0s10 = 10**(alpha*h0)
        h1s10 = 10**(alpha*h1)
        return np.log10(np.random.random(self.size)*(h1s10-h0s10) + h0s10) / alpha
        
    def H_to_mag(self):
        phase = np.arccos((self.r**2 + self.delta**2 - self.earth_dis**2) / (2 * self.r * self.delta))
        phase_integral = 2/3. * ((1-phase/np.pi)*np.cos(phase) + 1/np.pi*np.sin(phase))
        self.mag_g = self.H + 2.5 * np.log10((self.r**2 * self.delta**2) / phase_integral)
        self.mag_r = self.mag_g - 0.5
        self.mag_i = self.mag_r - 0.5
        self.mag_z = self.mag_i - 0.5
        
    def neptune_lambda(self):
        neptune = planets[8]
        ts = load.timescale()
        t = ts.tai(jd=self.mjd+2400000.500428) #37 leap seconds        
        self.x_n, self.y_n, self.z_n = neptune.at(t).ecliptic_position().au
        self.lambda_N = np.arctan2(self.y_n, self.x_n) % (2*np.pi)

    def kep_to_xyz(self, a, e, i, arg, node, M):
        # compute eccentric anomaly
        f = lambda E, M, e: E - e * np.sin(E) - M
        E0 = M
        E = newton(f, E0, args=(M, e))
        # compute true anomaly
        v = 2 * np.arctan2((1 + e)**0.5*np.sin(E/2.), (1 - e)**0.5*np.cos(E/2.))
        # compute the barycentric distance
        r = a * (1 - e*np.cos(E))
        # compute X,Y,Z
        X = r * (np.cos(node) * np.cos(arg + v) - np.sin(node) * np.sin(arg + v) * np.cos(i))
        Y = r * (np.sin(node) * np.cos(arg + v) + np.cos(node) * np.sin(arg + v) * np.cos(i))
        Z = r * (np.sin(i) * np.sin(arg + v))
        return X, Y, Z, r
        
    def xyz_to_equa(self, X0, Y0, Z0):
        earth = planets['earth']
        ts = load.timescale()
        t = ts.tai(jd=self.mjd+2400000.500428) #37 leap seconds
        epsilon =  23.43694 * np.pi/180.
        x_earth, y_earth, z_earth = earth.at(t).position.au
        self.earth_dis = (x_earth**2 + y_earth**2 + z_earth**2)**0.5
        X = X0 - x_earth
        Y = Y0 * np.cos(epsilon) - Z0 * np.sin(epsilon)  - y_earth
        Z = Y0 * np.sin(epsilon) + Z0 * np.cos(epsilon) - z_earth
        self.delta = (X**2 + Y**2+ Z**2)**0.5
        self.dec = np.arcsin(Z/(X**2+Y**2+Z**2)**0.5)
        self.ra = np.arctan2(Y, X) % (2*np.pi)


class plutino(Resonant):
    """
    generate plutinos
    units: au, radius
    
    """
    def __init__(self, size = 1000, mjd = 57023, e_c = 0.175, e_sigma = 0.06, i_sigma = 12, amp_c = 75, amp_max = 155, amp_min = 0, h0 = 0, h1 = 9.5):
        self.size = size
        self.mjd = mjd
        self.lambda_N = None
        self.neptune_lambda()
        self.mjd = mjd
        self.a = self.gen_a()
        self.e = self.gen_e(e_c, e_sigma)
        self.i = self.gen_i(i_sigma)
        self.amp = self.gen_amp(amp_c, amp_max, amp_min)
        self.phi = self.gen_phi(self.amp)
        self.M = self.gen_M()
        self.node = self.gen_node()
        self.arg = self.gen_arg()
        self.H = self.gen_H(h0, h1)
        cut = (self.e > 0) * (~np.isnan(self.i))
        self.a = self.a[cut]
        self.e = self.e[cut]
        self.i = self.i[cut]
        self.amp = self.amp[cut]
        self.phi = self.phi[cut]
        self.M = self.M[cut]
        self.node = self.node[cut]
        self.arg = self.arg[cut]
        self.H = self.H[cut]
        X, Y, Z, r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i,\
                                self.arg, self.node, self.M)) # r**2 = X**2 + Y**2 + Z**2
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Z = np.array(Z)
        self.r = np.array(r)
        self.xyz_to_equa(self.X, self.Y, self.Z)
        self.H_to_mag()
        
    
    def gen_a(self):
        return 39.45 + np.random.random(self.size) * 0.4 - 0.2
        
    def gen_phi(self, amp):
        return np.pi + amp * np.sin(2*np.pi*np.random.random(self.size)) 
        
    def gen_M(self):
        return 4*np.pi*np.random.random(self.size)
        
    def gen_arg(self):
        return (0.5*self.phi - 1.5*self.M - self.node + self.lambda_N) % (2*np.pi)
        

class trojan(Resonant):
    """
    generate trojans
    units: au, radius
    
    """
    def __init__(self, size = 1000, mjd = 57023, e_c = 0.04, e_sigma =0.04, i_sigma = 12, amp_c = 10, amp_max = 25, amp_min = 5, h0 = 0, h1 = 10):
        self.size = size
        self.mjd = mjd
        self.lambda_N = None
        self.neptune_lambda()
        self.a = self.gen_a()
        self.e = self.gen_e(e_c, e_sigma)
        self.i = self.gen_i(i_sigma)
        self.amp = self.gen_amp(amp_c, amp_max, amp_min)
        self.phi = self.gen_phi(self.amp)
        self.M = self.gen_M()
        self.node = self.gen_node()
        self.arg = self.gen_arg()
        self.H = self.gen_H(h0, h1)
        cut = (self.e > 0) * (~np.isnan(self.i))
        self.a = self.a[cut]
        self.e = self.e[cut]
        self.i = self.i[cut]
        self.amp = self.amp[cut]
        self.phi = self.phi[cut]
        self.M = self.M[cut]
        self.node = self.node[cut]
        self.arg = self.arg[cut]
        self.H = self.H[cut]
        X, Y, Z, r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i,\
                                self.arg, self.node, self.M)) # r**2 = X**2 + Y**2 + Z**2
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Z = np.array(Z)
        self.r = np.array(r)
        self.xyz_to_equa(self.X, self.Y, self.Z)
        self.H_to_mag()
        
    
    def gen_a(self):
        return 30.025 + np.random.random(self.size) * 0.25 - 0.125
            
    def gen_phi(self, amp):
        return ((1/3.*np.pi)+np.random.randint(-1, 1, self.size)*(2/3.*np.pi) + amp * np.sin(2*np.pi*np.random.random(self.size))) % (2*np.pi)    

    def gen_M(self):
        return 2*np.pi*np.random.random(self.size)
        
    def gen_arg(self):
        return (self.phi - self.M - self.node + self.lambda_N) % (2*np.pi)

class twotino(Resonant):
    """
    generate twotinos
    units: au, radius
    
    """
    def __init__(self, size = 1000, mjd = 57023, e_c = 0.2, e_sigma =0.06, i_sigma = 6, amp_c = 10, amp_max = 25, amp_min = 5, h0 = 0, h1 = 8):
        self.size = size
        self.mjd = mjd
        self.lambda_N = None
        self.neptune_lambda()
        self.phi0 = None
        self.gen_phi0()
        self.a = self.gen_a()
        self.e = self.gen_e(e_c, e_sigma)
        self.i = self.gen_i(i_sigma)
        self.amp = self.gen_amp()
        self.phi = self.gen_phi()
        self.M = self.gen_M()
        self.node = self.gen_node()
        self.arg = self.gen_arg()
        self.H = self.gen_H(h0, h1)
        cut = (~np.isnan(self.i))
        self.a = self.a[cut]
        self.e = self.e[cut]
        self.i = self.i[cut]
        self.amp = self.amp[cut]
        self.phi = self.phi[cut]
        self.M = self.M[cut]
        self.node = self.node[cut]
        self.arg = self.arg[cut]
        self.H = self.H[cut]
        X, Y, Z, r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i,\
                                self.arg, self.node, self.M)) # r**2 = X**2 + Y**2 + Z**2
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Z = np.array(Z)
        self.r = np.array(r)
        self.xyz_to_equa(self.X, self.Y, self.Z)
        self.H_to_mag()
        
    def gen_phi0(self):
        self.phi0 = np.random.randint(-1, 2, self.size)
        
    #def gen_e(self):
    #    return np.random.random(self.size) * 0.3 + 0.1 
    
    def gen_a(self):
        return 47.8 + np.random.random(self.size) * 0.4 - 0.2
            
    def gen_phi(self):
        phi = np.zeros(len(self.phi0))
        sym = (self.phi0 == 0)
        asym0 = (self.phi0 == -1)
        asym1 = (self.phi0 == 1)
        phi[sym] = np.pi + self.amp_sym * np.sin(2*np.pi*np.random.random(sym.sum())) % (2*np.pi)  
        phi[asym0] = -90 * np.pi/180. + self.amp_0 * np.sin(2*np.pi*np.random.random(asym0.sum())) % (2*np.pi) 
        phi[asym1] = 90 * np.pi/180. + self.amp_1 * np.sin(2*np.pi*np.random.random(asym1.sum())) % (2*np.pi) 
        return phi 
        
    def gen_amp(self):
        amp = np.zeros(len(self.phi0))
        sym = (self.phi0 == 0)
        asym0 = (self.phi0 == -1)
        asym1 = (self.phi0 == 1)
        self.amp_sym = (np.random.random(sym.sum())*15 + 135) * np.pi/180.
        self.amp_0 = (np.random.random(asym0.sum())*45 + 15) * np.pi/180.
        self.amp_1 = (np.random.random(asym1.sum())*45 + 15) * np.pi/180.
        amp[sym] = self.amp_sym
        amp[asym0] = self.amp_0
        amp[asym1] = self.amp_1
        return amp

    def gen_M(self):
        return 2*np.pi*np.random.random(self.size)
        
    def gen_arg(self):
        return (self.phi - 2 * self.M - self.node + self.lambda_N) % (2*np.pi)
        
        
def main():
    #p = plutino(size = 2000, e_c = 0.3, e_sigma = 0.01, amp_c = 1, amp_max = 2, amp_min = 0, i_sigma=12)
    p = plutino(size = 2000000, mjd=57023)
    #for i in zip(p.a, p.e, p.i, p.arg, p.node, p.M, p.H):
    #    print i
    print p.mag_g.min(), (p.mag_g < 23.5).sum()
    

if __name__ == '__main__':
    main()