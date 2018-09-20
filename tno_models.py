##########################################################################
#
# tno_models.py, version 0.3.2
#
# Generate resonant tno models 
#
# Author: 
# Edward Lin: hsingwel@umich.edu
#
# v0.1: trojan, plutino and twotino (close to complete) models
# v0.2: 4:1 toy model
# v0.3: modify trojan model
# v0.3.1: fix mean anomaly from radian to degree
# v0.3.2: fix q calculation 
##########################################################################


import numpy as np
from skyfield.api import Loader, Topos
from scipy.optimize import newton
load = Loader('./Skyfield-Data', expire=False)
planets = load('de430t.bsp')

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
        alpha_b = 0.9 # alpha should be a variable 
        alpha_f = 0.5
        hb = 8.3
        c = 3.2
        h0b = 10**(alpha_b*h0)
        hbb = 10**(alpha_b*hb)
        hbf = 10**(alpha_f*hb)
        h1f = 10**(alpha_f*h1)
        nb = (hbb - h0b)/alpha_b
        nf = 10**((alpha_b-alpha_f)*hb)/alpha_f * (h1f - hbf)/c     
        bfr = nb/(nb+nf)
        H_dis = np.random.random(self.size)
        bright = H_dis < bfr
        faint = H_dis >= bfr
        H_dis[bright] = np.log10(np.random.random(bright.sum())*(hbb-h0b) + h0b) / alpha_b
        H_dis[faint] = np.log10(np.random.random(faint.sum())*(h1f-hbf) + hbf) / alpha_f
        return H_dis
        
    def H_to_mag(self):
        phase = np.arccos((self.r**2 + self.delta**2 - self.earth_dis**2) / (2 * self.r * self.delta))
        # here should be changed to H-G model
        phase_integral = 2/3. * ((1-phase/np.pi)*np.cos(phase) + 1/np.pi*np.sin(phase))
        self.red = np.random.randint(2, size=self.size).astype('bool')
        self.mag_r = self.H + 2.5 * np.log10((self.r**2 * self.delta**2) / phase_integral)
        self.mag_g = np.copy(self.mag_r)
        self.mag_i = np.copy(self.mag_r)
        self.mag_z = np.copy(self.mag_r)
        self.mag_g[self.red] += 1.0
        self.mag_i[self.red] -= 0.5
        self.mag_z[self.red] -= 0.2
        self.mag_g[~self.red] += 0.6
        self.mag_i[~self.red] -= 0.2
        self.mag_z[~self.red] -= 0.05       
        self.gr = np.zeros(self.size)
        self.ri = np.zeros(self.size)
        self.iz = np.zeros(self.size)
        self.gr[self.red] += 1.0
        self.ri[self.red] += 0.5
        self.iz[self.red] += 0.2
        self.gr[~self.red] += 0.6
        self.ri[~self.red] += 0.2
        self.iz[~self.red] += 0.05

        
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
        epsilon =  23.43929 * np.pi/180. # obliquity
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
        self.size = len(self.H)
        X, Y, Z, r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, self.M)) # ecliptic Cartesian position at given epoch
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Z = np.array(Z)
        self.r = np.array(r)
        self.ra, self.dec, self.delta, self.earth_dis = self.xyz_to_equa(self.X, self.Y, self.Z, self.mjd) # Equatorial position at given epoch
        self.H_to_mag() # generate apparent magnitude 
        self.period = (self.a**3 * 4*np.pi**2/2.9630927492415936E-04)**0.5 # calculate orbital period
        self.peri_date = (self.mjd - (self.a**3 /2.9630927492415936E-04)**0.5 * self.M) + self.period # calculate next perihelion date
        X_peri, Y_peri, Z_peri, self.r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, np.zeros(len(self.a)))) # ecliptic Cartesian position at next perihelion date
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
    def __init__(self, size = 1000, mjd = 57023, e_c = 0.04, e_sigma =0.04, i_sigma = 26, amp_sigma = 15, h0 = 7, h1 = 10, L = None):
        self.size = size
        self.mjd = mjd
        self.lambda_N = None
        self.neptune_lambda()
        self.a = self.gen_a()
        self.e = self.gen_e(e_sigma)
        self.i = self.gen_i(i_sigma)
        self.amp = self.gen_amp(amp_sigma)
        self.phi = self.gen_phi(self.amp, L)
        self.M = self.gen_M()
        self.node = self.gen_node()
        self.arg = self.gen_arg()
        self.H = self.gen_H(h0, h1)
        cut = (self.e > 0) * (~np.isnan(self.i)) * (~np.isnan(self.amp)) * (self.i < 60*np.pi/180.)
        self.size = cut.sum()
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
        self.period = (self.a**3 * 4*np.pi**2/2.9630927492415936E-04)**0.5 # calculate orbital period
        self.peri_date = (self.mjd - (self.a**3 /2.9630927492415936E-04)**0.5 * self.M) + self.period # calculate next perihelion date
        X_peri, Y_peri, Z_peri, self.r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, np.zeros(len(self.a))))
        self.ra_peri, self.dec_peri, self.delta_peri, self.earth_dis_peri = self.xyz_to_equa(np.array(X_peri), np.array(Y_peri), np.array(Z_peri), self.peri_date)
        
    def gen_e(self, e_sigma):
        return np.random.rayleigh(scale=e_sigma, size = self.size)
        
    def gen_amp(self, amp_sigma):
        return np.arcsin(np.random.rayleigh(scale=amp_sigma * np.pi/180., size = self.size))

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
        self.period = (self.a**3 * 4*np.pi**2/2.9630927492415936E-04)**0.5 # calculate orbital period
        self.peri_date = (self.mjd - (self.a**3 /2.9630927492415936E-04)**0.5 * self.M) + self.period # calculate next perihelion date
        X_peri, Y_peri, Z_peri, self.r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, np.zeros(len(self.a))))
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
        
class fourone(Resonant):
    """
    generate four to one resonators
    units: au, radius
    
    """
    def __init__(self, size = 1000, mjd = 57023, i_sigma = 20, amp_c = 10, amp_max = 25, amp_min = 5, h0 = 0, h1 = 7):
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
        self.period = (self.a**3 * 4*np.pi**2/2.9630927492415936E-04)**0.5 # calculate orbital period
        self.peri_date = (self.mjd - (self.a**3 /2.9630927492415936E-04)**0.5 * self.M) + self.period # calculate next perihelion date
        X_peri, Y_peri, Z_peri, self.r = zip(*map(self.kep_to_xyz, self.a, self.e, self.i, self.arg, self.node, np.zeros(len(self.a))))
        self.ra_peri, self.dec_peri, self.delta_peri, self.earth_dis_peri = self.xyz_to_equa(np.array(X_peri), np.array(Y_peri), np.array(Z_peri), self.peri_date)
        
    def gen_phi0(self):
        # 1:1:1 leading/trailing/symmetry ratio, should be changed in future
        self.phi0 = np.random.randint(-1, 2, self.size)
        #self.phi0 = np.zeros(self.size) -1
    
    def gen_a(self):
        return 75.8 + np.random.random(self.size) * 0.0 - 0.0
            
    def gen_e(self):
        e = np.zeros(len(self.phi0))
        sym = (self.phi0 == 0)
        asym = (self.phi0 != 0)
        e[sym] = np.random.random(sym.sum())*0.0 + 0.6 # symmetry fourone have e between 0.1 and 0.35
        e[asym] = np.random.random(asym.sum())*0.0 + 0.6 # symmetry fourone have e between 0.1 and 0.4
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
        self.amp_sym = (np.random.random(sym.sum())*0 + 0) * np.pi/180.
        self.amp_0 = (np.random.random(asym0.sum())*0 + 0) * np.pi/180.
        self.amp_1 = (np.random.random(asym1.sum())*0 + 0) * np.pi/180.
        amp[sym] = self.amp_sym
        amp[asym0] = self.amp_0
        amp[asym1] = self.amp_1
        return amp

    def gen_M(self):
        return 2*np.pi*np.random.random(self.size)
        
    def gen_arg(self):
        return (self.phi - 4 * self.M - self.node + self.lambda_N) % (2*np.pi)


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
    ma = f.M * 180 / np.pi
    epoch_M = f.mjd - 15019.5
    H = f.H
    mag = f.mag_r
    sun_dist = f.r
    ra = f.ra * 180 / np.pi
    dec = f.dec * 180 / np.pi
    peri_date = f.peri_date - 15019.5
    peri_ra = f.ra_peri 
    peri_dec = f.dec_peri 
    r = f.r
    q = a * (1-e)
    gr = f.gr
    ri = f.ri
    iz = f.iz
    print('orbid,a,e,inc,omega,Omega,omega_bar,ma,epoch_M,H,mag,sun_dist,ra,dec,peri_date,peri_ra,peri_dec,r,gr,ri,iz')
    for i in range(len(a)):
        print('{0:07},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20}'.format(
                i+1,a[i],e[i],inc[i],omega[i], Omega[i], omega_bar[i],ma[i],epoch_M,H[i],mag[i],
                sun_dist[i],ra[i],dec[i],peri_date[i],peri_ra[i],peri_dec[i],r[i],gr[i],ri[i],iz[i]))


def output_oorb(fake_tnos):
    """
!!OID FORMAT q e i node argperi t_p H t_0 INDEX N_PAR MOID COMPCODE
10372163 COM 0.476075708311037E+01 0.758982208255032E-01 0.908354579162321E+01 0.551655067504993E+02 0.290407921221101E+03
0.501465855317433E+05 0.177200000000000E+02 0.493531606965741E+05 1 6 -0.100000000000000E+01 OPENORB 
    """
    f = fake_tnos
    a = f.a
    e = f.e
    inc = f.i
    omega = f.arg
    Omega = f.node
    omega_bar = (omega + Omega) % 360.
    ma = f.M
    epoch_M = f.mjd
    H = f.H
    mag = f.mag_r
    sun_dist = f.r
    ra = f.ra * 180 / np.pi
    dec = f.dec * 180 / np.pi
    peri_date = f.peri_date
    peri_ra = f.ra_peri * 180 / np.pi
    peri_dec = f.dec_peri * 180 / np.pi
    q = a*(1-e)
    print('!!OID FORMAT q e i node argperi t_p H t_0 INDEX N_PAR MOID COMPCODE')
    for i in range(len(a)):
        print('{0:08} COM {1} {2} {3} {4} {5} {6} {7} {8} 1 6 -0.100000000000000E+01 OPENORB'.format(i+1, q[i], e[i], inc[i], Omega[i], omega[i], peri_date[i], H[i], epoch_M))


def main():
    #p = plutino(size = 2000, e_c = 0.3, e_sigma = 0.01, amp_c = 1, amp_max = 2, amp_min = 0, i_sigma=12)
    #p = plutino(size = 2000, mjd=58423)
    t = trojan(size = 1000, mjd=58423, i_sigma = 26, L=4)
    #output_csv(t)
    output_oorb(t)

if __name__ == '__main__':
    main()
