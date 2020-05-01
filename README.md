# Segmented-Algorithm
Segmented Machine Learning Algorithm<p>

__What is better than one machine learning algorithm? Two or more machine learning algorithms!__<p>

### Summary<p>
Machine learning, more than learning through experience, is basically an optimization problem. In machine learning the objective is to minimize the error of different functions and algorithms.<p>
Data scientists look for ways to ‘generalize’ the best solution so that it can be applied to real life (test) data.
Depending on the application and its importance, sometimes 99% might not be enough (health applications, computer vision) and in some other cases 70% might be good enough (house price prediction).<p>
When dealing with a new classification or regression problem, a comparison between the most common models is often performed. The top 2 or 3 models, with minimum hyperparameter tuning and best accuracy (less error) are the ones that are usually taken to the final rounds until one model is selected and then optimized to its full capacity by tuning.<p>
What if the data scientist can use 2 or 3 different models to better predict one problem?<p>

### The need: Why Segmented Machine Learning Algorithm<p>
Image we are trying to solve a regression, classification or deep learning problem and one of all the features we have is a dummy variable such as yes/no (or male/female, etc.). Maybe some of the features are more important than others if this dummy variable is yes or no. Here is when SML algorithm can help the data scientist better predict an outcome.<p>
Continuing with this example, what if the data scientist can use a linear regression when the dummy variable is yes and use a lasso or ridge regression on the data when the value is no?<p>
Or if it is a classification problem, what if we can use Naïve Bayes for some data and Logistic Regression on the rest of the data?<p>

### How the Segmented Machine Learning Algorithm works<p>
The SML algorithm works relatively easy.<p>
1.	For each feature (columns in X), the algorithm orders the data in ascending order. It starts working with first feature X[:,0] going item by item before moving to the second feature X[:,1], etc.<p>
2.	Then it iterates for each different value inside current feature. Moving one by one, only skips repeated values of X.<p>
3.	At each iteration, the dataset is split in two. On one side called left, X (and its respective Y) are passed. Left filters data where X is less than or equal to the current value of X being analyzed. Right side is the rest of the data.<p>
a.	Example: If X.shape = (100,2) and Y.shape = (100,) and every value in the first feature of X is different from each other and every value of the second feature of X is different from each other, the algorithm would have 99+99 (198) iterations.<p>
b.	Same example as above, but the first feature is dummy and has only value of 0 and 1, assuming second feature of X are all different from each other. The algorithm would have 99+1 iterations (100). <p>
4.	Left values are ‘fit’ in the first machine learning object and the right values are ‘fit’ in the second object. The sum of both errors, in this case MSE, are added and compared in each iteration.<p>
5.	The algorithm keeps iterating until all values of X are analyzed and the information of the best value is stored in a dictionary (which is the output).<p>
6.	The algorithm can then be run recursively several times if needed.<p>

### Capabilities of the Algorithm<p>
SML algorithm works with regression and classification objects, technically it can work with deep learning objects but it might not be feasible to run it without modifications to improve speed.<p>
The algorithm works relatively fast if X doesn’t have a lot of different values (Ex. If it has some dummy features) and if the left and right machine learning objects are fast to execute.<p>
For example, if left and right objects are LinearRegression, it works faster than if both objects are LassoCV objects.<p>

The algorithm should work with all types of machine learning objects, the only requirement is that both sides (objects) are performing the same task (regression or classification). Just to give some examples:<p>
•	left side is LinearRegression and the right side LinearRegression.<p>
•	left side LogisticRegression and the right side LogisticRegression.<p>
•	left side LogisticRegression and the right side KNeighborsClassifier<p>
•	left side SVR and the right side SVR.<p>
•	left side SVM and the right side LogisticRegression.<p>

### Recursion<p>
Technically the model can be used recursively (it can be used to find the best split and then called again to find the best splits on each side and select the best one). This approach can be used to fit n-amount of different machine learning algorithms, for example use 4 linear regressions to solve a cubic function.<p>

### Limitations of the Algorithm<p>
The algorithm is not suited for slow machine learning objects that take long time to converge.<p>
In order to use it for classification problems, the only thing that would have to be changed is the ‘error’ function that better minimizes the error. Current algorithm evaluates MSE.<p>

### Future Work<p>
Things to be improved is speed for machine learning objects that might take long time to converge.<p>
For now, to use it in deep learning would be feasible only if the models are pre-trained.<p>
Error function. For now, the algorithms uses sklearn’s mse (from sklearn.metrics import mean_squared_error) but it can be easily adapted for other errors like for example mean absolute error, etc.<p>

### Prediction<p>
For the prediction, the best solution of the algorithm is required. The best solution is a dictionary with relevant information about the trained machine learning algorithms of the left and the right, also the value of X and its location (column).<p>
The prediction algorithm uses the fitted model of the left when values are less than or equal to the best value of X in the column with best split. For the other values, it predicts using the right fitted object.<p>

### Hyperparameter Tuning<p>
For each function on the left and on the side, there can be hyperparameter tuning. Although it is recommended to do it after running the SML algorithm.<p>

### Python Inputs and Outputs:<p>
INPUTS: <p>
	X: np.array with observations with shape (rows, columns). Can’t have NULLS or NAN values. Can’t be sparse.<p>
	Y: np.array with target value with shape (rows,)<p>
	Left: machine learning object. Example: Left = LinearRegression(n_jobs=-1)<p>
	Right: machine learning object. Example: Right = LinearRegression(n_jobs=-1)<p>
	Example: sml(x,y, left, right) => where left = LinearRegression(n_jobs=-1) and right = LinearRegression(n_jobs=-1)<p>

OUTPUT: dictionary with the following values:<p>
	best_r2: best r2 after all iterations<p>
	best_mse: best mse after all iterations<p>
	best_row: row location where best iteration happens<p>
	best_col: column location where best iteration happens<p>
	best_x: value of x where best iteration happens<p>
	left: fitted machine learning object with only the indexes where X <= best_x.<p>
	right: fitted machine learning object with only the indexes where X > best_x.<p>
Example: result = {'best_r2': 1.0, 'best_mse': 6.409494854920721e-32, 'best_row': 4, 'best_col': 0, 'best_x': 5, 'left': LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1, normalize=False), 'right': LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1, normalize=False)}<p>
This means that the minimum error was found when X[4:0] = 5. The left and right objects are the fit objects. Data from those objects can be extracted to get information such as coef_ and intercept_ if the object is a linear regression object.<p>
