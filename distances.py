
import numpy as np
from scipy import signal


def correlation(h1, h2):
    nh1 = np.sum(h1)/np.max(h1)
    nh2 = np.sum(h2)/np.max(h2)
    nh12 = np.sum(h1*h2)
    
    return nh12/(nh1 * nh2)
    
def chisquared_distance(h1, h2):
    distance = 0
    for i in range(0, len(h1)):
        if h1[i]+h2[i] != 0:
            distance += (h1[i]-h2[i])**2 / (h1[i]+h2[i])
    
    return distance/2
    

def percentrootmeansquare_distance(p1, p2):
    return (np.sum((p1 - p2)**2) / np.sum(p1**2))*100

# frechetdist 0.6
# Code adapted from: https://pypi.org/project/frechetdist/
def frechet_distance(p, q):
    def _c(ca, i, j, p, q):

        if ca[i, j] > -1:
            return ca[i, j]
        elif i == 0 and j == 0:
            ca[i, j] = np.linalg.norm(p[i]-q[j])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(ca, i-1, 0, p, q), np.linalg.norm(p[i]-q[j]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(ca, 0, j-1, p, q), np.linalg.norm(p[i]-q[j]))
        elif i > 0 and j > 0:
            ca[i, j] = max(
                min(
                    _c(ca, i-1, j, p, q),
                    _c(ca, i-1, j-1, p, q),
                    _c(ca, i, j-1, p, q)
                ),
                np.linalg.norm(p[i]-q[j])
                )
        else:
            ca[i, j] = float('inf')
        return ca[i, j]
    
    p = np.array(p)
    q = np.array(q)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')

    if len_p != len_q:
        raise ValueError('Input curves do not have the same dimensions.')

    ca = (np.ones((len_p, len_q)) * -1)

    dist = _c(ca, len_p-1, len_q-1, p, q)
    return dist


def mindistance(p1, gp2, distfunc):
    dist = np.zeros(len(gp2))
    for i in range(0,len(gp2)):
        dist[i] = distfunc(p1, gp2[i])    
    return np.min(dist)

def meanmindistance(gp1, gp2, distfunc):
    d = np.zeros(gp1.shape[0])
    for i in range(0, gp1.shape[0]):
        d[i] = distfunc(gp1[i]/np.max(gp1[i]), gp2[i]/np.max(gp2[i]))
        
    return np.mean(d), np.var(d)


def fwhm(n1):
    n1[0][0:len(n1[0])//2] = 0
    pos = np.argmax(n1[0])
    results_half = signal.peak_widths(n1[0], [pos], rel_height=0.5)
    fwhm = (results_half[3] - results_half[2]) * (n1[1][pos] - n1[1][pos-1])
    return fwhm
    
