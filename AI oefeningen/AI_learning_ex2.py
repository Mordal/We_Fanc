import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


path = 'C:\\Users\debie\Documents\\AI_\ipython-notebooks-master\data\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

data_mean = data2.mean()
data_std = data2.std()
data = data2


data2 = (data2 - data2.mean()) / data2.std()



def computeCost(X, y, theta):

    #de inner matrix berekenen om de cost te bepalen
    
    inner = np.power(((X*theta.T)-y),2)

    #sum van alle waarden in matrix, gedeeld door 2x het aantal rijen in X
    return np.sum(inner) / (2*len(X))

def gradientDescent(X, y, theta, alpha, iters):

    #maakt een matrix met shape =theta gevuld met 0'en
    temp = np.matrix(np.zeros(theta.shape))
    
    #ravel zet alle waarden in 1 rij: creeert kolommen bij
    #shape[1]: geeft het aantel kolomen
    #parameters == het totaal aantal waarden in theta (= rijen*kolomen)
    parameters = int(theta.ravel().shape[1])

    # cost =  array van 0'en. iters stuks
    cost = np.zeros(iters)

       
    for i in range(iters):
        #.T functie: draait rijen en kolommen om
        #error = ((97,2)*(2,1))-(97,1) ->uitgedrukt in '.shape'
        #feitelijk: X[0,0]*theta.T[0,0] + X[0,1]*theta.T[1,0] ..zo per rij
        error = (X*theta.T)-y

        #j = 0 of 1
        #for j in range(parameters):
            #term = np.multiply(error, X[:,j])
            #temp[0,j] = theta[0,j] - ((alpha / len(X))*np.sum(term))
        temp[0,:] = [theta[0,j] - ((alpha / len(X))*np.sum(np.multiply(error, X[:,j]))) for j in range(parameters)]
        
                        
            
        theta = temp
        cost[i] = computeCost(X,y,theta)
        
      
    return theta, cost

data2.insert(0,'Ones',1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

alpha = 0.01
iters = 1000

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
# get the cost (error) of the model
print(str(computeCost(X2,y2,g2)))



fig, ax = plt.subplots(figsize=(4,4))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

fig.show()


def prijs(size, bedrooms):
    size_leveld = (size - data_mean[0]) / data_std[0]
    bedr_leveld = (bedrooms - data_mean[1]) / data_std[1]
    
    berek_matr = np.matrix([[1,size_leveld,bedr_leveld]])
    
    kost_matr = g2 * berek_matr.T
    kost = kost_matr[0,0] * data_std[2] + data_mean[2]

    return kost




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

line = np.linspace(data.values[:,0].min(),data.values[:,0].max(),50)

yss = np.zeros(shape=(5,line.shape[0]))
xss = np.zeros(shape=(5,line.shape[0]))
zss = np.zeros(shape=(5,line.shape[0]))


#tel van 5->1
for yr in range(5):
    yss[yr,:] = yr+1
    xss[yr,:] = line
    zss[yr,:] = [prijs(xss[yr,l],yss[yr,l]) for l in range(line.shape[0])]
    

    #for l in range(line.shape[0]):
     #   zss[yr,l] = prijs(xss[yr,l],yss[yr,l])
        

xs = data.values[:,0]
ys = data.values[:,1]
zs = data.values[:,2]
ax.scatter(xs, ys, zs, c='r', marker='o')
ax.plot_wireframe(xss, yss, zss)

ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

plt.show()


