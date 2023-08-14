from ast import MatMult
from tkinter import font
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys
#import the given dataset
WS=pd.read_csv('Dataset.csv' ,header=None,index_col=False)
m=np.array(WS)
col,row=len(m),len(m)
column=[[1 for i in range(1)] for i in range(row)]
#adding a new column to store the cluster number for each dataset
m=np.column_stack((m,column))

colours={0:'lightcoral',1:'orchid',2:'greenyellow',3:'turquoise',4:'purple'}

#function to initialize with k clusters randomly from given dataset
def cluster(k):
    cluster=[]
    for i in range(k):
        mean=np.random.randint(0,len(m)-1)
        cluster.append(m[mean,:])
    return cluster
    #cluster contains the mean of the respective clusters

#function for the first assignment of each datapoints to the randomly initialized clusters
def assign(cluster,k):
    for i in range(len(m)):
        pt_x= m[i,:]
        min=sys.maxsize
        for j in range(k):
            pt_xx=cluster[j][0]
            pt_yy=cluster[j][1]
            if math.sqrt((pt_x[0]-pt_xx)**2+(pt_x[1]-pt_yy)**2) < min:
                min=math.sqrt((pt_x[0]-pt_xx)**2+(pt_x[1]-pt_yy)**2)
                m[i][2]=j       


#function to compute error w.r.t. to iterations
def compute_error(cluster,k):
    min=0
    for i in range(len(m)):
        pt_x= m[i,:]
        for j in range(k):
            if(m[i][2]==j):
                pt_xx=cluster[j][0]
                pt_yy=cluster[j][1]
                min+=math.sqrt((pt_x[0]-pt_xx)**2+(pt_x[1]-pt_yy)**2)
    return min      

#function to reassign datapoints to clusters based on the closest mean
def reassignment(cluster,k):
    #flag variable to keep track of convergence
    flag=False
    for i in range(len(m)):
        pt_x= m[i,:]
        z=m[i][2]
        min=sys.maxsize 
        for j in range(k):
            pt_xx=cluster[j][0]
            pt_yy=cluster[j][1]
            if min> math.sqrt((pt_x[0]-pt_xx)**2+(pt_x[1]-pt_yy)**2):
                min=math.sqrt((pt_x[0]-pt_xx)**2+(pt_x[1]-pt_yy)**2)
                z=j 
        if(m[i][2]!=z):
            m[i][2]=z
            flag=True
    return flag


#function to calculate mean after each reassignment of datapoint in different clusters
def meancalculation(cluster,k):
    meanx=[]
    meany=[]
    sumx=[]
    sumy=[]
    count=[]
    for i in range(k):
        meanx.append(0)
        meany.append(0)
        sumx.append(0)
        sumy.append(0)
        count.append(0)
    for i in range(len(m)):
        if(m[i][2]==0):
            sumx[0]+=m[i][0]
            sumy[0]+=m[i][1]
            count[0]+=1
            meanx[0]=sumx[0]/count[0]
            meany[0]=sumy[0]/count[0]
        elif(m[i][2]==1):
            sumx[1]+=m[i][0]
            sumy[1]+=m[i][1]
            count[1]+=1
            meanx[1]=sumx[1]/count[1]
            meany[1]=sumy[1]/count[1]
        elif(m[i][2]==2):
            sumx[2]+=m[i][0]
            sumy[2]+=m[i][1]
            count[2]+=1
            meanx[2]=sumx[2]/count[2]
            meany[2]=sumy[2]/count[2]
        elif(m[i][2]==3):
            sumx[3]+=m[i][0]
            sumy[3]+=m[i][1]
            count[3]+=1
            meanx[3]=sumx[3]/count[3]
            meany[3]=sumy[3]/count[3]
    for i in range(k):  
        cluster[i][0]=meanx[i]
        cluster[i][1]=meany[i]
    return cluster

#funtion to print vornoi regions w.r.t. different clusters
def printvornoi(cluster,k):
    f3=plt.figure(3)
    a=[]
    b=[]
    c=[]
    i=-11
    z=0
    while(i<11.0):
        j=-11
        while(j<11.0):
            pt_x=i
            pt_y=j
            min = sys.maxsize 
            for l in range(k):
                pt_xx=cluster[l][0]
                pt_yy=cluster[l][1]
                if min> math.sqrt((pt_x-pt_xx)**2+(pt_y-pt_yy)**2):
                    min=math.sqrt((pt_x-pt_xx)**2+(pt_y-pt_yy)**2)
                    z=l
            a.append(i)
            b.append(j)
            c.append(z)
            j=j+0.1
        i=i+0.1
    
    v=pd.DataFrame(list(zip(a,b,c)),columns=['X','Y','Z'])
    plt.xlabel('X-axis',color='black')
    plt.ylabel('Y-axis',color='black')
    plt.title('Vornoi Regions',color='black')
    plt.scatter(v['X'][(v.Z==0)],v['Y'][v.Z==0],marker='o',color=colours[0])
    plt.scatter(v['X'][(v.Z==1)],v['Y'][v.Z==1],marker='o',color=colours[1])
    plt.scatter(v['X'][(v.Z==2)],v['Y'][v.Z==2],marker='o',color=colours[2])
    plt.scatter(v['X'][(v.Z==3)],v['Y'][v.Z==3],marker='o',color=colours[3])
    plt.scatter(v['X'][(v.Z==4)],v['Y'][v.Z==4],marker='o',color=colours[4])

#function to implement k-means clustering algorithm
def mainfunc(clusters,k):
    count=0
    iteration=[]
    total_error=[]
    sum1=assign(clusters,k)
    while(True):
        flag=0
        clusters=meancalculation(clusters,k)
        count+=1
        sum1=compute_error(clusters,k)
        total_error.append(sum1)
        iteration.append(count)
        flag =reassignment(clusters,k)
        if(flag==False):
            break
    
    printvornoi(clusters,k)
    f1=plt.figure(1)
    for j in range(k):
        for i in range (len(m)):
            if (m[i][2]==j):
                plt.xlabel('X-axis',color='black')
                plt.ylabel('Y-axis',color='black')
                plt.title('clusters for fixed initialization ',color='black')
                plt.scatter(m[i][0],m[i][1],c=colours[j])
    plt.grid()
    #plotting error w.r.t. iterations 
    f2=plt.figure(2)
    plt.xlabel('X-axis --Iterations',color='black')
    plt.ylabel('Y-axis --Total_error',color='black')
    plt.title('Error Function w.r.t. Iterations',color='black')
    plt.plot(iteration,total_error) 
    plt.grid()
    plt.show()

#fixing a random initialization
k=5
clusters=cluster(k)
#run the algorithm for 2,3,4,5 for above fixed initialization
for i in range(2,6):
    mainfunc(clusters,i)