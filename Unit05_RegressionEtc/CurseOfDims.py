#Illustrates the curse of dimensionality. For each dimension n from 1 to 20,
#draws 1000 vectors from the standard multivariate normal (mean zero, identity
#covariance), takes their Euclidean magnitudes, and plots the median and some
#quantiles of the magnitudes against n. Points do not stay near the origin as
#n grows: nearly all of them end up at about the same, growing, distance from
#it, so "near" and "far" lose their meaning for distance-based methods like knn.
#
#Python translation of CurseOfDims.R (Claude, 2026-10-15), kept as parallel to
#the R as possible. Needs numpy and matplotlib. Run with the working directory
#set to this folder. Saves CurseOfDims_py.pdf so as not to overwrite the R pdf.

import numpy as np
import matplotlib.pyplot as plt

rng=np.random.default_rng(101)

nmax=20
nvecs=1000
probs=[0.025,0.25,0.5,0.75,0.975]

#One row per n, one column per quantile
res=np.full((nmax,len(probs)),np.nan)
for n in range(1,nmax+1):
    #nvecs by n matrix of independent N(0,1) draws: each row is one vector
    #from the standard multivariate normal in n dimensions
    x=rng.standard_normal((nvecs,n))
    mags=np.sqrt(np.sum(x**2,axis=1))
    res[n-1,:]=np.quantile(mags,probs)
np.set_printoptions(precision=2,suppress=True)
print("   "+"  ".join("q"+str(p) for p in probs))
print(np.round(res,2))

#Plot
ns=np.arange(1,nmax+1)
fig,ax=plt.subplots(figsize=(6,4.5))
ax.plot(ns,res[:,2],"o-",color="black",label="Median")
ax.plot(ns,res[:,1],"--",color="black",label="25th and 75th percentiles")
ax.plot(ns,res[:,3],"--",color="black")
ax.plot(ns,res[:,0],":",color="black",label="2.5th and 97.5th percentiles")
ax.plot(ns,res[:,4],":",color="black")
ax.set_ylim(0,res.max())
ax.set_xlabel("Dimension, n")
ax.set_ylabel("Magnitude of a standard normal vector")
ax.set_title("1000 random vectors per n")
ax.legend(loc="upper left",frameon=False)
fig.savefig("CurseOfDims_py.pdf")
plt.show()
