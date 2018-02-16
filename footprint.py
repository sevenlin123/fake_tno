from __future__ import division

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import ephem
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Polygon

def define_footprint(polydef='round13-poly.txt', ecliptic_plots=False):
    ra = []
    dec = []
    
    with open(polydef) as f:
        for line in f:
            if line[0] != '#':
                s = line.split(' ')
                ra_i = -999
                for i in range(len(s)):
                    if s[i] != '' and ra_i==-999:
                        ra_i = float(s[i])
                        ra.append(ra_i*np.pi/180)
                    elif s[i] != '' and ra_i != -999:
                        dec.append(float(s[i])*np.pi/180)
        ec_lat = []
        ec_lon = []
        for i in range(len(ra)):
            ec = ephem.Equatorial(ra[i], dec[i])
            eq = ephem.Ecliptic(ec)
            ec_lon.append(eq.lon)
            ec_lat.append(eq.lat)
    if ecliptic_plots is True:
        lon = []
        for i in range(len(ec_lon)):
            if ec_lon[i]>ephem.degrees('180:00:00'):
                lon.append(ec_lon[i]-2*np.pi)
            else: lon.append(ec_lon[i])
        return np.array(lon)*180/np.pi, np.array(ec_lat)*180/np.pi
    else:
        return np.array(ra)*180/np.pi, np.array(dec)*180/np.pi

def SNfields():
    C1 = ephem.readdb("C1,f,03:37:05.83,-27:06:41.8,23.0,2000")
    C2 = ephem.readdb("C2,f,03:37:05.83,-29:05:18.2,23.0,2000")
    C3 = ephem.readdb("C3,f,03:30:35.62,-28:06:00.0,24.0,2000")
    X1 = ephem.readdb("X1,f,02:17:54.17,-04:55:46.2,23.0,2000")
    X2 = ephem.readdb("X2,f,02:22:39.48,-06:24:43.6,23.0,2000")
    X3 = ephem.readdb("X3,f,02:25:48.00,-04:36:00.0,24.0,2000")
    S1 = ephem.readdb("S1,f,02:51:16.80, 00:00:00.0,23.0,2000")
    S2 = ephem.readdb("S2,f,02:44:46.66,-00:59:18.2,23.0,2000")
    E1 = ephem.readdb("E1,f,00:31:29.86,-43:00:34.6,23.0,2000")
    E2 = ephem.readdb("E2,f,00:38:00.00,-43:59:52.8,23.0,2000")
    fields = [C1, C2, C3, X1, X2, X3, S1, S2, E1, E2]
    for f in fields:
        f.compute()
    return fields

def DECamEllipse(ra, dec, rotation=0, facecolor='g', alpha=0.5):
    # An approximation to the DECam field of view, suitable e.g. for plotting
    semimajor_deg = 180/np.pi*ephem.degrees('1.08')
    semiminor_deg = 180/np.pi*ephem.degrees('0.98')
    center = (180/np.pi*ra, 180/np.pi*dec)
    return patches.Ellipse(center, 2*semimajor_deg, 2*semiminor_deg, rotation, facecolor=facecolor, alpha=alpha)


 
def SNpatches(fields, ecliptic_plots=False):
    SNellipse=[]
    if ecliptic_plots is True:
        for f in fields:
            ecl = ephem.Ecliptic(ephem.Equatorial(f.a_ra, f.a_dec))
            lon = ecl.lon if ecl.lon<ephem.degrees('180') else ecl.lon-2*np.pi
            SNellipse.append(DECamEllipse(lon, ecl.lat, rotation=18.5))
    else:   
        SNellipse = [DECamEllipse(f.a_ra, f.a_dec) for f in fields]
#    print len(SNellipse)
    return SNellipse


def buildmap(ecliptic_plots=True):
    lon, lat = define_footprint(ecliptic_plots=ecliptic_plots) 
#    lon2, lat2 = define_footprint(polydef='poly_bliss_p9.txt', ecliptic_plots=ecliptic_plots) 
    m = Basemap(lon_0=0,
	            projection='moll', celestial=True)
    x, y = m( lon, lat )
#    x2,y2 = m(lon2, lat2)
    xy = zip(x,y)
#    xy2 = zip(x2,y2)
    foot = patches.Polygon( xy, facecolor='cornflowerblue', edgecolor=None, alpha=0.4 )
#    foot2 = patches.Polygon( xy2, facecolor='lightpink', edgecolor=None, alpha=0.4 )
    plt.gca().add_patch(foot)
#    plt.gca().add_patch(foot2)
    fields = SNfields()
    for f in fields:
		if ecliptic_plots:
			ecl = ephem.Ecliptic(ephem.Equatorial(f.a_ra, f.a_dec))
			lon = ecl.lon if ecl.lon<ephem.degrees('180') else ecl.lon-2*np.pi    
			m.tissot(lon*180/np.pi, ecl.lat*180/np.pi, 1.05, 100, facecolor='g', alpha=0.5)
		else:
			ra = f.a_ra if f.a_ra<ephem.degrees('180') else f.a_ra-2*np.pi
			m.tissot(ra*180/np.pi, f.a_dec*180/np.pi, 1.05, 100, facecolor='g', alpha=0.5)
    m.drawmapboundary()
    parallels = np.arange(-180.,181,20.)
    m.drawparallels(parallels,labels=[False,True,True,False], alpha=0.4)
    meridians = np.arange(-180.,181.,20.)
    m.drawmeridians(meridians, alpha=0.4)
    return m


def main():
	plt.figure(1, facecolor='w', edgecolor='k', figsize=(16,8))
	m = buildmap(ecliptic_plots=False)
	plt.savefig('footprint-equatorial.pdf')
	plt.show()

if __name__== '__main__':
	main()
