# Segmented-Algorithm
Segmented Machine Learning Algorithm<p>
<p>
What is better than one machine learning algorithm? Two or more machine learning algorithms!<p>
<p>
### Summary<p>
Machine learning, more than learning through experience, is basically an optimization problem. In machine learning the objective is to minimize the error of different functions and algorithms.<p>
Data scientists look for ways to ‘generalize’ the best solution so that it can be applied to real life (test) data.
Depending on the application and its importance, sometimes 99% might not be enough (health applications, computer vision) and in some other cases 70% might be good enough (house price prediction).<p>
When dealing with a new classification or regression problem, a comparison between the most common models is often performed. The top 2 or 3 models, with minimum hyperparameter tuning and best accuracy (less error) are the ones that are usually taken to the final rounds until one model is selected and then optimized to its full capacity by tuning.<p>
What if the data scientist can use 2 or 3 different models to better predict one problem?<p>
<p>
#### The need: Why Segmented Machine Learning Algorithm<p>
Image we are trying to solve a regression, classification or deep learning problem and one of all the features we have is a dummy variable such as yes/no (or male/female, etc.). Maybe some of the features are more important than others if this dummy variable is yes or no. Here is when SML algorithm can help the data scientist better predict an outcome.<p>
Continuing with this example, what if the data scientist can use a linear regression when the dummy variable is yes and use a lasso or ridge regression on the data when the value is no?<p>
Or if it is a classification problem, what if we can use Naïve Bayes for some data and Logistic Regression on the rest of the data?<p>
<p>
