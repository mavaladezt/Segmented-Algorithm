# Segmented Machine Learning Algorithm<p>

__What is better than one machine learning algorithm? Two or more machine learning algorithms!__<p>

### Summary<p>
Machine learning is an optimization problem. The objective is to minimize the error of different functions and algorithms.<br/>
Data scientists look for ways to ‘generalize’ the best solution so that it can be applied to real-life (test) data.<br/>
Depending on the application and its importance, sometimes 99% might not be enough (health applications, computer vision) and in some other cases, 70% might be good enough (house price prediction).

![Image description](../master/image02.jpg)

### The need: Why Segmented Machine Learning Algorithm<p>
Some regression problems can be solved with more precision if more than one model is implemented. In these cases, SML works by dividing the data into ‘sets’ and apply different models to predict the target variable.

### How the Segmented Machine Learning Algorithm works<p>
The SML algorithm works in theory very similar to a decision tree. Still, instead of finding differences in the target label, it runs two models at each point of the data and splits where the error of the total error (of both models) is the minimum.<br/>
1.	For each feature (columns in X) and value, the algorithm evaluates the best place to do a split by fitting two models, one on the ‘left’ of the data and the other on the ‘right’. Both models make a prediction of the target variable and the errors are added. The algorithm ‘records’ the best split, where the sum of errors if the minimum.<br/>
2.	The model runs several times until the recursion level is met.

### Speed<p>
SML works well with algorithms that converge fast such as Linear Regression.<br/>
To improve speed in large datasets, the algorithm has the option of analyzing data in percentiles. This implementation can analyze datasets with millions of values in a couple of seconds.<br/>
The hyperparameter for controlling speed is called max_iterations_per_col. For example, if max_iterations_per_col is equal to 100, for each feature with more than 100 different values the algorithm is going only to evaluate 100. If it has less than 100 it will iterate at every value of that particular feature.<br/>
To increase ‘precision,’ a bigger number can be taken, but it will slow down the algorithm since it will now evaluate more options.

### Limitations of the Algorithm<p>
SML algorithm works with regression and to work with classification a few changes would need to be made. For example, when ‘fitting’ both classification algorithms on the right and left, the algorithm would have to send both positive and negative cases to both sides so that the classification algorithms can run.<br/>
Error Function: The current algorithm evaluates MSE, but different error functions and calculations can be easily changed.

### Differences between SML and Piecewise<p>
SML is not piecewise. Instead of defining a function as multiple sub-functions like piecewise, SML finds regression (of selected model) approximations to functions for different ranges and dimensions of x.

### How to Run the Algorithm with Different Parameters and Models (algorithm only implemented in python)
__Function:
sml(X_train,y_train, model, max_levels=np.inf, max_iterations_per_col=100, count=0)__

Run SML with Linear Regression, 3 levels of recursion and default iterations per column = 100:
sml(X_train,y_train,LinearRegression(),3,100)

(faster than previous) Run SML with Linear Regression, 3 levels of recursion and default iterations per column = 50:
sml(X_train,y_train,LinearRegression(),3,50)

Run SML with Lasso:
sml(X_train,y_train,SVR())

Run SML with Support Vector Regression, 8 levels of recursion and max iterations of 200:
sml(X_train,y_train,SVR(),8,200)

