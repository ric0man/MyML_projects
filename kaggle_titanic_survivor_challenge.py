#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
train_data=pd.read_csv(r"C:\Users\Rohan\titanictrain.csv")
train_data.head()


# In[2]:


drop_columns=["PassengerId","Name","Ticket","Cabin","Embarked"]
#train_data.info()
clean_trdata= train_data.drop(drop_columns,axis=1)
clean_trdata.head(n=5)


# In[3]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
clean_trdata["Sex"]=le.fit_transform(clean_trdata["Sex"])
clean_trdata.head()


# In[4]:


clean_trdata=clean_trdata.fillna(clean_trdata["Age"].mean())
clean_trdata.head(n=23)


# In[5]:


clean_trdata.loc[1]


# In[6]:


output_col=["Survived"]
input_col=["Pclass","Sex","Age","SibSp","Parch","Fare"]
X=clean_trdata[input_col]
Y=clean_trdata[output_col]
print(X.shape,Y.shape)
print(type(X))


# In[7]:


def entropy(col):
    counts=np.unique(col,return_counts=True)
    N=float(col.shape[0])
    #print(counts)
    #print(N)
    ent=0.0
    for ix in counts[1]:
        p=ix/N
        ent+=(-1*p*np.log2(p))
    return ent


# In[8]:


col=np.array([1,0,1,0,1,1,1,0])
entropy(col)


# In[9]:


def divide_data(x_data,fkey,fval):
    x_right=pd.DataFrame([],columns=x_data.columns)
    x_left=pd.DataFrame([],columns=x_data.columns)
    
    for ix in range(x_data.shape[0]):
        val=x_data[fkey].loc[ix]
        if val>fval:
            x_right=x_right.append(x_data.loc[ix])
        else:
            x_left=x_left.append(x_data.loc[ix])
            
    return x_left,x_right
    
    


# In[10]:


def information_gain(x_data,fkey,fval):
    left,right=divide_data(x_data,fkey,fval)
    #total functions on left and right 
    l=float(left.shape[0]/x_data.shape[0])
    r=float(right.shape[0]/x_data.shape[0])
    #all examples come to one side!
    if left.shape[0]==0 or right.shape[0]==0:
        return -1000000
    i_gain=entropy(x_data.Survived)-(l*entropy(left.Survived)+r*entropy(right.Survived))
    return i_gain


# In[11]:


#Testing our function
for fx in X.columns:
    print(fx)
    print(information_gain(clean_trdata,fx,clean_trdata[fx].mean()))


# In[12]:


class DecisionTree:
    #Constructor
    def __init__(self,depth=0,max_depth=5):
        self.left=None
        self.right=None
        self.fkey=None
        self.fval=None
        self.max_depth=max_depth
        self.depth=depth
        self.target=None
        
        
    def train(self,X_train):
        features=['Pclass','Sex','Age','SibSp','Parch','Fare']
        info_gains=[]
        for ix in features:
            i_gain=information_gain(X_train,ix,X_train[ix].mean())
            info_gains.append(i_gain)
        self.fkey=features[np.argmax(info_gains)]
        self.fval=X_train[self.fkey].mean()
        print("Making Tree feature is",self.fkey)
        
        #split data
        data_left,data_right=divide_data(X_train,self.fkey,self.fval)
        data_left=data_left.reset_index(drop=True)
        data_right=data_right.reset_index(drop=True)
        
        #Truly a leaf node
        if data_left.shape[0]==0 or data_right.shape[0]==0:
            if X_train.Survived.mean()>=0.5:
                self.target="Survive"
            else:
                self.target="Dead"
            return
        #stop early when depth>=max depth
        if(self.depth>=self.max_depth):
            if X_train.Survived.mean()>=0.5:
                self.target="Survive"
            else:
                self.target="Dead"
            return
        #recurssive case
        self.left=DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(data_left)
        self.right=DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(data_right)
        
        #we can target at every point 
        if X_train.Survived.mean()>=0.5:
                self.target="Survive"
        else:
                self.target="Dead"
        return
        #making predictions
    def predict(self,test):
   
        if test[self.fkey]>self.fval:
         #go to right 
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
             if self.left is None:
                return self.target
             return self.left.predict(test)


# In[13]:


d=DecisionTree()
#d.train(clean_trdata)


# In[14]:



#Train validation test set split
split=int(0.7*clean_trdata.shape[0])
train_data=clean_trdata[:split]
test_data=clean_trdata[split:]
test_data=test_data.reset_index(drop=True)
print(train_data.shape,test_data.shape)
dt=DecisionTree()
dt.train(train_data)


# In[15]:


#making predictions
y_pred=[]
for ix in range(test_data.shape[0]):
    y_pred.append(dt.predict(test_data.loc[ix]))
print(y_pred)
print("jai mata di")
y_actual=test_data[output_col]
print(y_actual)


# In[16]:


le=LabelEncoder()
y_pred=le.fit_transform(y_pred)
print(y_pred)


# In[17]:


y_pred=np.array(y_pred).reshape((-1,1))
print(y_actual.shape,y_pred.shape)


# In[18]:


acc=np.sum(np.array(y_actual)==np.array(y_pred))/y_pred.shape[0]
print(acc)


# In[19]:


#Decision trees using sci-kit learn
from sklearn.tree import DecisionTreeClassifier as dtc
sk_tree=dtc(criterion='entropy',max_depth=5)
sk_tree.fit(train_data[input_col],train_data[output_col])
sk_tree.predict(test_data[input_col])
sk_tree.score(test_data[input_col],test_data[output_col])



# In[21]:


#decision tree visualisation using graphviz 
import pydotplus
import matplotlib.pyplot as plt
#from PIL import Image
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

dot_data=StringIO()
export_graphviz(sk_tree,out_file=dot_data,filled=True,rounded=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[ ]:





# In[ ]:




