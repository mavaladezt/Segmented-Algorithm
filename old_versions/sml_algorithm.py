
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import r2_score
from copy import deepcopy
import time

#data = np.genfromtxt('data.csv', delimiter=',')
#data = data[1:,:]
#x = data[:,0:3]
#y = data[:,-1]

#x = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5]).reshape(-1,1)
#y = np.array([26,17,10,5,2,1,2,5,10,17,26])
#x = np.concatenate((x,np.ones(11).reshape(-1,1)),axis=1)


#=====DATA0============================================
data = np.genfromtxt('Advertising.csv', delimiter=',')
data = data[1:,1:]
x = data[:,0:3]
y = data[:,-1]

#=====DATA1============================================
#x=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
#y=np.array([0.5,1.5,2.5,3.5,4.5,16,17,18,19,20])

#=====DATA2============================================
#x=np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5]).reshape(-1,1)
#y=np.array([26,17,10,5,2,1,2,5,10,17,26])

#=====DATA3============================================
#x=np.array([1,2,3,4,5,6,7,8,9,10,11]).reshape(-1,1)
#y=np.array([0,0.301029995663981,0.477121254719662,0.602059991327962,0.698970004336019,-0.778151250383644,-0.845098040014257,-0.903089986991944,-0.954242509439325,-1,-1.04139268515822])

#=====DATA4============================================
#x=np.array([-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5]).reshape(-1,1)
#y=np.array([3.125,-2,-4.125,-4,-2.375,0,2.375,4,4.125,2,-3.125])

lr = LinearRegression(n_jobs=-1).fit(x,y)

print("r^2:",r2_score(y, lr.predict(x)))
print("MSE:",mean_squared_error(y, lr.predict(x)))

left = LinearRegression(n_jobs=-1)
right = LinearRegression(n_jobs=-1)

#fig = plt.figure('Plot Data + Regression')
#ax1 = fig.add_subplot(111)
#ax1.plot(x, y, marker='x', c='b', label='data')
#ax1.plot(x,lr.predict(x), marker='o',c='g', label='linear r.')
#ax1.set_xlabel('x')
#ax1.set_ylabel('y')
#ax1.set_title('Data vs Regression\nr2: '+str( round((r2_score(y, lr.predict(x))),3) )+'  MSE: '+ str(round((mean_squared_error(y, lr.predict(x))),3)))
#ax1.legend(loc=2)
#plt.show()

def sml(X,Y, left, right):
    '''
    sml algorithm inputs features (x) and target (y) and find best position to split 2 different machine learning
      objects. Both objects have to perform same type of prediction (classification or regression)
    Input: 
      X: numpy features dataset with shape (rows,columns)
      Y: numpy target variable with shape (rows,)
      left: machine learning object 1 (Ex. sklearn.linear_model.LinearRegression object)
      right: machine learning object 2 (Ex. sklearn.linear_model.LinearRegression object)
    Output: dictionary with the following keys
      best_r2: best r2 after all iterations
      best_mse: best mse after all iterations
      best_row: row location where best iteration happens
      best_col: column location where best iteration happens
      best_x: value of x where best iteration happens
      left: fitted machine learning object with only the indexes where X <= best_x.
      right: fitted machine learning object with only the indexes where X > best_x.

    '''
    result = {'best_r2':None,'best_mse':np.inf,'best_row':None,'best_col':None,'best_x':None,'left':None,'right':None}
    for j in range(X.shape[1]):
        y = Y[X[:,j].argsort()]
        x = X[X[:,j].argsort()]
        for i in range(x.shape[0]-1):
            if x[i,j] == x[i+1,j]:
                continue
            left.fit(x[:i+1,:],y[:i+1])
            right.fit(x[i+1:,:],y[i+1:])
            left_pred = left.predict(x[:i+1,:])
            right_pred = right.predict(x[i+1:,:])
            error = mean_squared_error(y, np.concatenate((left_pred,right_pred)))
            if error < result['best_mse']:
                result['best_r2'] = r2_score(y, np.concatenate((left_pred,right_pred)))
                result['best_mse'] = error
                result['best_row'] = i
                result['best_col'] = j
                result['best_x'] = x[i,j]
                result['left'] = deepcopy(left)
                result['right'] = deepcopy(right)
    return result

def sml_predict(x,result):
    l=0
    r=0
    y_pred = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if(x[i,result['best_col']]) > result['best_x']:
            y_pred[i] = result['right'].predict(x[i,:].reshape(1,-1))
            r+=1
        else:
            y_pred[i] = result['left'].predict(x[i,:].reshape(1,-1))
            l+=1
    return y_pred

result = sml(x,y, left, right)
print(result)

y_pred = sml_predict(x,result)

print("MSE:",mean_squared_error(y, y_pred))
print("r^2:",r2_score(y, y_pred))



#======================




start_time = time.time()
lr = LinearRegression(n_jobs=-1).fit(x,y)
print("r^2:",r2_score(y, lr.predict(x)))
print("MSE:",mean_squared_error(y, lr.predict(x)))
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
result = sml(x,y, left, right)
#print(result)
print("--- %s seconds ---" % (time.time() - start_time))



x=np.array([-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5]).reshape(-1,1)
y=np.array([3.125,-2,-4.125,-4,-2.375,0,2.375,4,4.125,2,-3.125])

x=x[0:5,:]
x.shape
y=y[0:5]
result = sml(x,y, left, right)
print(result)
y_pred = sml_predict(x,result)
l1=y_pred[:]

x=np.array([-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5]).reshape(-1,1)
y=np.array([3.125,-2,-4.125,-4,-2.375,0,2.375,4,4.125,2,-3.125])
x=x[5:,:]
y=y[5:]
result = sml(x,y, left, right)
print(result)
y_pred = sml_predict(x,result)
l2=y_pred[:]

print("r^2:",r2_score(y, y_pred))
print("MSE:",mean_squared_error(y, y_pred))

fig = plt.figure('Integrated Plot')
ax1 = fig.add_subplot(111)
ax1.plot(x, y, marker='x', c='b', label='data')
ax1.plot(x,y_pred, marker='o',c='r', label='proposed')
ax1.plot(x,lr.predict(x), marker='o',c='g', label='linear r.')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Data vs Regression vs Approach\nr2: '+str( round(( r2_score(y, y_pred) ),3) )+'  MSE: '+ str(round(( mean_squared_error(y, y_pred) ),3)))
plt.legend(loc=2);
plt.show()

