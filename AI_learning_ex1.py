import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:\\Users\debie\Documents\\AI_\ipython-notebooks-master\data\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

print("Data-head values: \n"+ str(data.head()))
print("Data-description: \n" + str(data.describe()))

data.plot(kind='scatter', x='Population', y='Profit', figsize=(4,4))
#plt.show()

def computeCost(X, y, theta):

    #de inner matrix berekenen om de cost te bepalen
    
    inner = np.power(((X*theta.T)-y),2)

    #sum van alle waarden in matrix, gedeeld door 2x het aantal rijen in X
    return np.sum(inner) / (2*len(X))



data.insert(0, 'Ones', 1)

cols = data.shape[1]

#print('cols: ' + str(cols))

X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

#print('X: ' + str(X))
#print('y: ' + str(y))

#convert from data frame to numpy array
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

print('theta: ' + str(theta))
#X.shape, theta.shape, y.shape
#print('X: ' + str(X))
#print('y: ' + str(y))


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
         #   term = np.multiply(error, X[:,j])
          #  temp[0,j] = theta[0,j] - ((alpha / len(X))*np.sum(term))

        #bovenstaande loop in 1 keer doen:
        temp[0,:] = [theta[0,j] - ((alpha / len(X))*np.sum(np.multiply(error, X[:,j]))) for j in range(parameters)]
        




        
        theta = temp
        cost[i] = computeCost(X,y,theta)
    return theta, cost

alpha = 0.01
iters = 1000

g, cost = gradientDescent(X,y, theta, alpha, iters)

#PLOT results
x = np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0,0] + (g[0,1]*x)

fig, ax = plt.subplots(figsize=(4,4))
ax.plot(x,f,'r',label = 'Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

fig.show()

fig, ax = plt.subplots(figsize=(4,4))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

fig.show()


from sklearn import linear_model  
model = linear_model.LinearRegression()  
model.fit(X, y)


x = np.array(X[:, 1].A1)  
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')
fig.show()
