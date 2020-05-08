import numpy as np
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split

from copy import deepcopy
import time

#***** SML ALGORITHM *****

def variance(y,y_pred):
    #Calculate error of prediction.
    #Input:
    #  y: values of y for comparing the prediction
    #  y_pred: predicted values
    #Output:
    #  error value calculation
    return mean_squared_error(y,y_pred)


def sqsplit(X_train, y_train, model, max_iterations_per_col):
    '''
    Function evaluates where to split using the variance function
    Input:
        X_train: X training set (numpy)
        y_train: y training set (target vector)
        model: machine learning object with '.fit' properties
		max_iterations_per_col. Max numbers of iterations per column (handy with big datasets)
    #Output:
        feature or column where data splits the best
        cut: cut-value of the best cut
        bestvar: variance where best cut happens
    '''
    N,D = X_train.shape
    bestvar = np.inf
    feature = np.inf
    cut = np.inf
    lr_left = deepcopy(model)
    lr_right = deepcopy(model)
    for i in range(D):
        values = list(set(X_train[:,i]))
        values.remove(max(values))
        values.sort()
        iters = len(values)
        if iters > max_iterations_per_col:
            percentiles = (np.array(range(1,max_iterations_per_col))/max_iterations_per_col*iters).astype(int)
            values = np.array(values)[percentiles]
        for value in values:
            msk = X_train[:,i]<= value
            lr_left.fit(X_train[msk],y_train[msk])
            lr_right.fit(X_train[~msk],y_train[~msk])
            var = variance(np.concatenate((y_train[msk],y_train[~msk])),np.concatenate((lr_left.predict(X_train[msk]),lr_right.predict(X_train[~msk]))))
            if var <= bestvar:
                bestvar = var
                cut = value
                feature = i
#    print(feature, cut, bestvar)
    return feature, cut, bestvar


class TreeNode(object):
    #Tree class: class to save the split information
    def __init__(self, left, right, feature, cut, prediction):
        self.left = left
        self.right = right
        self.feature = feature
        self.cut = cut
        self.prediction = prediction

def sml(X_train,y_train,model,max_levels=np.inf,max_iterations_per_col=100,count=0):
    '''
    sml function that will be called recursively
    Input:
      X_train: feature numpy matrix
      y_train: target vector
      model: machine learning object with '.fit' properties
      max_levels: how many recursive levels to go do in each split
      max_iterations_per_column
      count: used to calculate levels of 'recursiveness'. Must always start with 0.
    Output:
      tree: decision tree
	'''

    n,d = X_train.shape
    indices_in_range_n = np.arange(n)
    lr = deepcopy(model)
    lr.fit(X_train,y_train)
    
    prediction = deepcopy(lr.fit(X_train,y_train))
    
    min_size = len(y_train)
    if np.all(y_train == y_train[0]) or count>=max_levels or len(y_train)<=10:
        return TreeNode(None, None, None, None, prediction)    #this would be a leaf
    else:
        #this would be a branch
        feature, cut, bestvar = sqsplit(X_train, y_train, model,max_iterations_per_col)     #find new place where to split        
        left_indices =  indices_in_range_n[X_train[:, feature] <= cut]       #left side
        right_indices =  indices_in_range_n[X_train[:, feature] > cut]       #right side
        left_leaf = sml(X_train[left_indices,:],y_train[left_indices],model,max_levels,max_iterations_per_col,count+1)      #find new branch/leaf
        right_leaf = sml(X_train[right_indices,:],y_train[right_indices],model,max_levels,max_iterations_per_col,count+1)  #find new branch/leaf
        tree = TreeNode(left_leaf, right_leaf, feature, cut, prediction)     #construct the tree
        left_leaf.parent = tree
        right_leaf.parent = tree
        return tree

def predictions(tree,X_test):
    '''
    calculate predictions based on sml tree prediction
    Input:
      tree: TreeNode decision information
      X_test:  n x d numpy matrix of data points
    Output:
      pred: n-dimensional vector of predictions (target)
	'''
    n,d = X_test.shape
    pred = np.zeros(n)    #empty prediction vector

    for i in range(n):
        current_node = tree
        while True:                 #infinite loop until it reaches a leaf
            if current_node.left == current_node.right == None:   #if both sides are None it is a leaf
                current_predictor = current_node.prediction                
                pred[i] = current_predictor.predict(X_test[i,:].reshape(-1,d))
                break               #found solution, so break to continue with next i
            else:
                if X_test[i,current_node.feature] <= current_node.cut:
                    current_node = current_node.left
                else:
                    current_node = current_node.right
    return pred



#***** STEP 1: SELECT DATASET *****

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
#x = np.random.uniform(low=-10, high=10, size=(5000,1))
#y = (-x**3+5*x).reshape(-1,)       #y = -x3 + 5x

#=====DATA5============================================
#from sklearn.datasets import make_regression
#np.random.seed(1)
#x, y = make_regression(n_samples=10000, n_features=10, noise=50, random_state=1)

#=====DATA6============================================
#x=np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5]*1000).reshape(-1,1)
#y=np.array([26,17,10,5,2,1,2,5,10,17,26]*1000)

#=====DATA7============================================
#np.random.seed(1)
#x = np.random.rand(2000)
#x=np.concatenate((x,-x))
#y = x**2
#x=x.reshape(-1,1)

#=====DATA8============================================
from sklearn.datasets import load_boston
x, y = load_boston(return_X_y=True)

#=====DATA9============================================
#from sklearn.datasets import load_diabetes
#x, y = load_diabetes(return_X_y=True)

#=====DATA10===========================================
#x = np.random.uniform(low=-10, high=10, size=(5000,1))
#y = np.cos(x).reshape(-1,)

#=====DATA11===========================================
#x = np.random.uniform(low=-10, high=10, size=(100000,1))
#y = np.cos(x).reshape(-1,)
#y = y+(x[:,0]>=0)*5



#***** STEP 2: TRAIN TEST SPLIT *****

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)




#***** STEP 3: TRAIN MODEL AND TEST IT *****




#Baseline

print('Baseline')
start_time = time.time()
lr = LinearRegression(n_jobs=-1).fit(X_train,y_train)
print("--- %s seconds ---" % round((time.time() - start_time),4))
print('\tTrain:')
print('\t\tr2: ',r2_score(y_train,lr.predict(X_train)))
print('\t\tMSE: ',mean_squared_error(y_train,lr.predict(X_train)))
print('\tTest:')
print('\t\tr2: ',r2_score(y_test,lr.predict(X_test)))
print('\t\tMSE: ',mean_squared_error(y_test,lr.predict(X_test)),'\n')

# fig = plt.figure('Plot',figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
# ax1 = fig.add_subplot(111)
# ax1.scatter(x, y, c='b', s=3, label='data')
# ax1.scatter(x,lr.predict(x), c='g', s=3, label='linear r.')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('')
# ax1.legend(loc=2)
# plt.show()

print('SML (LinearRegression)')
start_time = time.time()
parent = sml(X_train,y_train,LinearRegression(n_jobs=-1),2,100)
print("--- %s seconds ---" % round((time.time() - start_time),4))
pred_train = predictions(parent,X_train)
pred_test = predictions(parent,X_test)
pred_total=predictions(parent,x)
print('\tTrain:')
print('\t\tr2: ',r2_score(y_train,pred_train))
print('\t\tMSE: ',mean_squared_error(y_train,pred_train))
print('\tTest:')
print('\t\tr2: ',r2_score(y_test,pred_test))
print('\t\tMSE: ',mean_squared_error(y_test,pred_test))


# lr = LinearRegression(n_jobs=-1).fit(x,y)
# fig = plt.figure('Plot',figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
# ax1 = fig.add_subplot(111)
# ax1.scatter(x, y, c='b', s=3, label='data')
# ax1.scatter(x,lr.predict(x), c='g', s=3, label='linear r.')
# ax1.scatter(x, pred_total, c='r', s=3, label='sml')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('')
# ax1.legend(loc=2)
# plt.show()







# t=[0,0,0]

# start1 = time.time()
# parent = sml(X_train,y_train,LinearRegression(n_jobs=-1),2,100)
# t[0]=(time.time() - start1)

# start2 = time.time()
# parent = sml(X_train,y_train,LinearRegression(n_jobs=-1),2,100)
# t[1]=(time.time() - start2)

# start3 = time.time()
# parent = sml(X_train,y_train,LinearRegression(n_jobs=-1),2,100)
# t[2]=(time.time() - start3)

# print('Avg. Time: ',round(np.mean(t),2))
# print(t)



