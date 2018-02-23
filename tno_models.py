##########################################################################
#
# tno_models.py, version 0.1
#
# Generate resonant tno models 
#
# Author: 
# Edward Lin: hsingwel@umich.edu
#
# v0.1: trojan, plutino and twotino (close complete) models
#
##########################################################################


import numpy as np
from skyfield.api import load, Topos
from scipy.optimize import newton
planets = load('de422.bsp')


class Resonant:
    """
    The parent resonant class
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
        # single power-law luminosity function
        alpha = 0.9 # alpha should be a variable 
        h0s10 = 10**(alpha*h0)
        h1s10 = 10**(alpha*h1)
        return np.log10(np.random.random(self.size)*(h1s10-h0s10) + h0s10) / alpha
        
    def H_to_mag(self):
        phase = np.arccos((self.r**2 + self.delta**2 - self.earth_dis**2) / (2 * self.r * self.delta))
        # here should be changed to H-G model
        phase_integral = 2/3. * ((1-phase/np.pi)*np.cos(phase) + 1/np.pi*np.sin(phase))
        # assume g-r, r-i, i-z all equal to 0.5
        self.mag_r = self.H + 2.5 * np.log10((self.r**2 * self.delta**2) / phase_integral)
        self.mag_g = self.mag_r + 0.5
        self.mag_i = self.mag_r - 0.5
        self.mag_z = self.mag_i - 0.5
        
    def neptune_lambda(self):
        neptune = planets[8]
        ts = load.timescale()
        t = ts.tai(jd=self.mjd+2400000.500428) #37 leap seconds        
        self.x_n, self.y_n, self.z_n = neptune.at(t).ecliptic_position().au
        self.lambda_N = np.arctan2(self.y_n, self.x_n) % (2*np.pi)

    def kep_to_xyz(self, a, e, i, arg, node, M):
        # compute eccentric anomaly E
        f = lambda E, M, e: E - e * np.sin(E) - M
        E0 = M
        E = newton(f, E0, args=(M, e))
        # compute true anomaly v
        v = 2 * np.arctan2((1 + e)**0.5*np.sin(E/2.), (1 - e)**0.5*np.cos(E/2.))
        # compute the barycentric distance r
        r = a * (1 - e*np.cos(E))
        # compute X,Y,Z
        X = r * (np.cos(node) * np.cos(arg + v) - np.sin(node) * np.sin(arg + v) * np.cos(i))
        Y = r * (np.sin(node) * np.cos(arg + v) + np.cos(node) * np.sin(arg + v) * np.cos(i))
        Z = r * (np.sin(i) * np.sin(arg + v))
        return X, Y, Z, r
        
    def xyz_to_equa(self, X0, Y0, Z0, epoch):
        earth = planets['earth']
        ts = load.timescale()
        t = ts.tai(jd=epoch+2400000.500428) #37 leap seconds
        epsilon =  23.43694 * np.pi/180. # obliquity
        x_earth, y_earth, z_earth = earth.at(t).position.au # earth IRCS position
        earth_dis = (x_earth**2 + y_earth**2 + z_earth**2)**0.5
        # transfer ecliptic to IRCS and shift to Geocentric
        X = X0 - x_earth 
        Y = Y0 * np.cos(epsilon) - Z0 * np.sin(epsilon)  - y_earth
        Z = Y0 * np.sin(epsilon) + Z0 * np.cos(epsilon) - z_earth
        # Cartesian to spherical coordinate
        delta = (X**2 + Y**2+ Z**2)**0.5
        dec = np.arcsin(Z/(X**2+Y**2+Z**2)**0.5)
        ra = np.arctan2(Y, X) % (2*np.pi)
        return ra, dec, delta, earth_dis

class plutino(Resonant):
    """
    generate plutinos
    units: au, radius
    
    """
    def __init__(self, size = 1000, mjd = 57023, e_c = 0.175, e_sigma = 0.06, i_sigma = 12, amp_c = 75, amp_max = 155, amp_min = 0, h0 = 0, h1 = 8.5):
        self.size = size # sample size of model 
        self.mjd = mjd # epoch
        self.lambda_N = None 
        self.neptune_lambda() # calculate mean longitude of neptune at given epoch
        # generate a, e, i, amp, phi, M, node, arg and H distribution 
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
        self.M = self.M[cut] % (2*np.pi)
        self.node = self.node[cut]
        self.arg = self.arg[cut]
        self.H = self.H[cut]
        X, Y, Z, r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, self.M)) # ecliptic Cartesian position at given epoch
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Z = np.array(Z)
        self.r = np.array(r)
        self.ra, self.dec, self.delta, self.earth_dis = self.xyz_to_equa(self.X, self.Y, self.Z, self.mjd) # Equatorial position at given epoch
        self.H_to_mag() # generate apparent magnitude 
        self.period = (self.a**3 * 4*np.pi**2/0.000295912208)**0.5 # calculate orbital period
        self.peri_date = (self.mjd - (self.a**3 /0.000295912208)**0.5 * self.M) + self.period # calculate next perihelion date
        X_peri, Y_peri, Z_peri, self.q = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, np.zeros(len(self.a)))) # ecliptic Cartesian position at next perihelion date
        self.ra_peri, self.dec_peri, self.delta_peri, self.earth_dis_peri = self.xyz_to_equa(np.array(X_peri), np.array(Y_peri), np.array(Z_peri), self.peri_date) # Equatorial position at next perihelion date
        
    
    def gen_a(self):
        return 39.45 + np.random.random(self.size) * 0.4 - 0.2
        
    def gen_phi(self, amp):
        return np.pi + amp * np.sin(2*np.pi*np.random.random(self.size)) 
        
    def gen_M(self):
        # for p:q resonant, the M should between 0 to q x 2pi
        return 4*np.pi*np.random.random(self.size)
        
    def gen_arg(self):
        return (0.5*self.phi - 1.5*self.M - self.node + self.lambda_N) % (2*np.pi)
        

class trojan(Resonant):
    """
    generate trojans
    units: au, radius
    
    """
    def __init__(self, size = 1000, mjd = 57023, e_c = 0.04, e_sigma =0.04, i_sigma = 12, amp_c = 10, amp_max = 25, amp_min = 5, h0 = 0, h1 = 9.5, L = None):
        self.size = size
        self.mjd = mjd
        self.lambda_N = None
        self.neptune_lambda()
        self.a = self.gen_a()
        self.e = self.gen_e(e_c, e_sigma)
        self.i = self.gen_i(i_sigma)
        self.amp = self.gen_amp(amp_c, amp_max, amp_min)
        self.phi = self.gen_phi(self.amp, L)
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
        self.M = self.M[cut] % (2*np.pi)
        self.node = self.node[cut]
        self.arg = self.arg[cut]
        self.H = self.H[cut]
        X, Y, Z, r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, self.M)) # r**2 = X**2 + Y**2 + Z**2
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Z = np.array(Z)
        self.r = np.array(r)
        self.ra, self.dec, self.delta, self.earth_dis = self.xyz_to_equa(self.X, self.Y, self.Z, self.mjd)
        self.H_to_mag()
        self.period = (self.a**3 * 4*np.pi**2/0.000295912208)**0.5 # calculate orbital period
        self.peri_date = (self.mjd - (self.a**3 /0.000295912208)**0.5 * self.M) + self.period # calculate next perihelion date
        X_peri, Y_peri, Z_peri, self.q = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, np.zeros(len(self.a))))
        self.ra_peri, self.dec_peri, self.delta_peri, self.earth_dis_peri = self.xyz_to_equa(np.array(X_peri), np.array(Y_peri), np.array(Z_peri), self.peri_date)
        
    def gen_a(self):
        return 30.025 + np.random.random(self.size) * 0.25 - 0.125
            
    def gen_phi(self, amp, L):
        # 1:1 leading/trailing ratio
        if L == 4:
            return ((1/3.*np.pi) + amp * np.sin(2*np.pi*np.random.random(self.size))) % (2*np.pi)  
        if L == 5:
            return ((-1/3.*np.pi) + amp * np.sin(2*np.pi*np.random.random(self.size))) % (2*np.pi)  
        else:
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
    def __init__(self, size = 1000, mjd = 57023, i_sigma = 6, amp_c = 10, amp_max = 25, amp_min = 5, h0 = 0, h1 = 7):
        self.size = size
        self.mjd = mjd
        self.lambda_N = None
        self.neptune_lambda()
        self.phi0 = None
        self.gen_phi0()
        self.a = self.gen_a()
        self.e = self.gen_e()
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
        self.M = self.M[cut] % (2*np.pi)
        self.node = self.node[cut]
        self.arg = self.arg[cut]
        self.H = self.H[cut]
        X, Y, Z, r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, self.M)) # r**2 = X**2 + Y**2 + Z**2
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Z = np.array(Z)
        self.r = np.array(r)
        self.ra, self.dec, self.delta, self.earth_dis = self.xyz_to_equa(self.X, self.Y, self.Z, self.mjd)
        self.H_to_mag()
        self.period = (self.a**3 * 4*np.pi**2/0.000295912208)**0.5 # calculate orbital period
        self.peri_date = (self.mjd - (self.a**3 /0.000295912208)**0.5 * self.M) + self.period # calculate next perihelion date
        X_peri, Y_peri, Z_peri, self.q = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, np.zeros(len(self.a))))
        self.ra_peri, self.dec_peri, self.delta_peri, self.earth_dis_peri = self.xyz_to_equa(np.array(X_peri), np.array(Y_peri), np.array(Z_peri), self.peri_date)
        
    def gen_phi0(self):
        # 1:1:1 leading/trailing/symmetry ratio, should be changed in future
        self.phi0 = np.random.randint(-1, 2, self.size)
    
    def gen_a(self):
        return 47.8 + np.random.random(self.size) * 0.4 - 0.2
            
    def gen_e(self):
        e = np.zeros(len(self.phi0))
        sym = (self.phi0 == 0)
        asym = (self.phi0 != 0)
        e[sym] = np.random.random(sym.sum())*0.25 + 0.1 # symmetry twotinos have e between 0.1 and 0.35
        e[asym] = np.random.random(asym.sum())*0.3 + 0.1 # symmetry twotinos have e between 0.1 and 0.4
        return e

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
        
def output_csv(fake_tnos):
    """
    output model to csv file
    
    """
    f = fake_tnos
    a = f.a
    e = f.e
    inc = f.i * 180 / np.pi
    omega = f.arg * 180 / np.pi
    Omega = f.node * 180 / np.pi
    omega_bar = (omega + Omega) % 360.
    ma = f.M
    epoch_M = f.mjd - 15019.5
    H = f.H
    mag = f.mag_r
    sun_dist = f.r
    ra = f.ra * 180 / np.pi
    dec = f.dec * 180 / np.pi
    peri_date = f.peri_date - 15019.5
    peri_ra = f.ra_peri * 180 / np.pi
    peri_dec = f.dec_peri * 180 / np.pi
    q = f.q
    print('orbid,a,e,inc,omega,Omega,omega_bar,ma,epoch_M,H,mag,sun_dist,ra,dec,peri_date,peri_ra,peri_dec,q')
    for i in range(len(a)):
        print('{0:07},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17}'.format(
                i+1,a[i],e[i],inc[i],omega[i], Omega[i], omega_bar[i],ma[i],epoch_M,H[i],mag[i],
                sun_dist[i],ra[i],dec[i],peri_date[i],peri_ra[i],peri_dec[i],q[i]))
        
def main():
    #p = plutino(size = 2000, e_c = 0.3, e_sigma = 0.01, amp_c = 1, amp_max = 2, amp_min = 0, i_sigma=12)
    #p = plutino(size = 10, mjd=57023)
    t = trojan(size = 5000)
    output_csv(t)
    

if __name__ == '__main__':
    main()