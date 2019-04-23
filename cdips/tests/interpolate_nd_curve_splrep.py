from scipy.interpolate import splprep, splev

import numpy as np
import matplotlib.pyplot as plt

# make ascending spiral in 3-space
t= np.linspace(0,1.75*2*np.pi,100)

x = np.sin(t)
y = np.cos(t)
z = t

# add noise
x+= np.random.normal(scale=0.1, size=x.shape)
y+= np.random.normal(scale=0.1, size=y.shape)
z+= np.random.normal(scale=0.1, size=z.shape)

# spline parameters
s=3.0 # smoothness parameter
k=2 # spline order
nest=-1 # estimate of number of knots needed (-1 = maximal)

# find the knot points
tckp,u = splprep([x,y,z],s=s,k=k,nest=-1)

# evaluate spline, including interpolated points
xnew,ynew,znew = splev(np.linspace(0,1,400),tckp)

plt.subplot(2,2,1)
data,=plt.plot(x,y,'bo-',label='data')
fit,=plt.plot(xnew,ynew,'r-',label='fit')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(2,2,2)
data,=plt.plot(x,z,'bo-',label='data')
fit,=plt.plot(xnew,znew,'r-',label='fit')
plt.legend()
plt.xlabel('x')
plt.ylabel('z')

plt.subplot(2,2,3)
data,=plt.plot(y,z,'bo-',label='data')
fit,=plt.plot(ynew,znew,'r-',label='fit')
plt.legend()
plt.xlabel('y')
plt.ylabel('z')

plt.savefig('../results/fitting_experiments/splprep_demo.png')
