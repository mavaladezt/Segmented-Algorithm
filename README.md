# Segmented Machine Learning Algorithm<p>

__What is better than one machine learning algorithm? Two or more machine learning algorithms!__<p>

### Summary<p>
Machine learning is an optimization problem, the objective is to minimize the error of different functions and algorithms.<p>
Data scientists look for ways to ‘generalize’ the best solution so that it can be applied to real life (test) data.<p>
Depending on the application and its importance, sometimes 99% might not be enough (health applications, computer vision) and in some other cases 70% might be good enough (house price prediction).


###The need: Why Segmented Machine Learning Algorithm<p>
Some regression problems can be solved with more precision if more than one model is implemented. In these cases, SML works perfectly dividing the data in ‘sets’ and apply different models to predict the target variable.

###How the Segmented Machine Learning Algorithm works<p>
The SML algorithm works in theory very similar to a decision tree but instead of finding differences in the target label, it runs 2 models at each point of the data and splits where the error of the total error (of both models) is the minimum.<p>
1.	For each feature (columns in X) and value, the algorithm evaluates the best place to make a split by fitting 2 models, one on the ‘left’ of the data and the other on the ‘right’. Both models make a prediction of the target variable and the errors are summed. The algorithm ‘records’ the best split, where the sum of errors if the minimum.<p>
2.	Model runs several times until the recursion level is met.

###Speed<p>
SML works fast with algorithms that converge really fast such as Linear Regression.<p>
To improve speed in large datasets, the algorithm has the option of analyzing data in percentiles instead. This implementation can analyze datasets with millions of values in a couple of seconds.<p>

###Limitations of the Algorithm<p>
The algorithm works algorithm is not suited for slow machine learning objects that take long time to converge.<p>
SML algorithm works with regression and in order to work with classification a few changes would need to be made. For example, when ‘fitting’ both classification algorithms on the right and left, the algorithm would have to send both positive and negative cases to both sides so that the classification algorithms are able to run.<p>
Error Function: Current algorithm evaluates MSE but different error functions and calculations can be easily changed.

###Differences between SML and Piecewise<p>
SML is not piecewise. Instead of defining a function as multiple sub-functions like piecewise, SML finds linear (regression) approximations to functions for different ranges and dimensions of x.
