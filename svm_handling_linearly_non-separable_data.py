#!/usr/bin/env python
# coding: utf-8

# In[10]:


from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
X,Y=make_circles(n_samples=500,noise=0.05)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show(block=True)


# In[11]:


def phi(X):
    """non linear transformation"""
    X1=X[:,0]
    X2=X[:,1]
    X3=X1**2+X2**2
    X_=np.zeros((X.shape[0],3))
    X_[:,:-1]=X
    X_[:,-1]=X3
    return X_
X_=phi(X)
    


# In[12]:


def plot3d(X,show=True):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')
    X1=X[:,0]
    X2=X[:,1]
    X3=X[:,2]
    ax.scatter(X1,X2,X3,zdir='z',s=20,c=Y,depthshade=True)
    if(show==True):
        plt.show(block=True)
    return ax
Z=plot3d(X_)    


# In[13]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='lbfgs')
acc2d=cross_val_score(lr,X,Y,cv=5).mean()
print("Accuracy X(2D) is %.4f"%(acc2d*100))
acc3d=cross_val_score(lr,X_,Y,cv=5).mean()
print("Accuracy X(3D) is % .4f"%(acc3d*100))


# In[14]:


lr.fit(X_,Y)
wts=lr.coef_
bias=lr.intercept_
m,n=np.meshgrid(range(-2,2),range(-2,2))
#print(m,n)
z=-(wts[0,0]*m+wts[0,1]*n+bias)/wts[0,2]
#print(z)
ax=plot3d(X_,False)
ax.plot_surface(m,n,z,alpha=0.2)
plt.show(block=True)


# In[ ]:




