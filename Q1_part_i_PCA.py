from tkinter import font
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import the given dataset
WS=pd.read_csv('Dataset.csv' ,header=None,index_col=False)
m=np.array(WS)

#datacentering
mnew=m-np.mean(m,axis=0)
#transpose of centered data
mnewt=mnew.T

#cov will store the product of mnewt and mnew
cov=[[0,0],[0,0]]
origin=[0,0]
for i in range(len(mnewt)):
    for j in range(len(mnew[0])):
        for k in range(len(mnew)):
            cov[i][j] += mnewt[i][k] * mnew[k][j]

#cov matrix successfully created

#calculating eigen values and eigen vectors for cov matrix
eval, evec= np.linalg.eig(cov)

#sort eigenvalues
sorted_index=np.argsort(eval)[::-1]
sorted_eval=eval[sorted_index]
sorted_evec=evec[:,sorted_index]
for i in range(2):
    sorted_eval[i]=sorted_eval[i]/1000

#printing the eigen values and eigen vectors of cov matrix
print("Eigen_value = " ,sorted_eval)
print("Eigen_vector = " ,sorted_evec)

#calculating variance along each eigen vector
for i in range(2):
    variance=sorted_eval[i]/sorted_eval.sum()
    print("variance = " ,round(variance.real*100,3),"%")


#plot both the eigen vectors along with dataset
plt.scatter(mnew[:,0],mnew[:,1],c='orchid')
plt.xlabel('X-axis',color='black')
plt.ylabel('Y-axis',color='black')
plt.title('PCA on Centered Dataset',color='black')
plt.axline((0,0),(-0.323516 , -0.9462227),color='greenyellow',label='PC1')
plt.axline((0,0),(-0.9462227,0.323516),color='orange',label='PC2')
plt.legend()
plt.grid()
plt.show()

