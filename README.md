# Project 3

Nitu Girish Mohan

In this project, I will be looking at the Gradient Boosting Algorithm, testing the Boosted Locally Weighted Regressor on different datasets/kernels, and using complete KFold cross validations to compare with other regressors.


## Gradient Boosting
Let's start by discussing Gradient Boosting, a boosting technique in machine learning that operates on the principle of minimizing the total prediction error by combining the best possible subsequent model with the previous models. The crucial aspect is to establish target outcomes for the next model to minimize the error. This approach generates a predictive model composed of a collection of weak prediction models, usually decision trees. When a decision tree functions as the weak learner, the resulting algorithm is known as gradient-boosted trees, and it frequently outperforms random forest (which we'll demonstrate later in the code).
 
Assume you have an regressor $F$ and, for the observation $x_i$ we make the prediction $F(x_i)$. To improve the predictions, we can regard $F$ as a 'weak learner' and therefore train a decision tree (we can call it $h$) where the new output is $y_i-F(x_i)$. So, the new predictor is trained on the residuals of the previous one. Thus, there are increased chances that the new regressor

$$\large F + h$$ 

is better than the old one, $F.$

Main task: implement this idea in an algorithm and test it on real data sets.

The algorithm follows the order of the structure below:

<img src="GBDiagram.png" class="LR" alt=""> 

To implement the Gradient Boosting algorithm with user defined choices for Regressor_1 and Regressor_2 we will do the following. First we'll start with the boosted regressor implementation from class.

```python
def boosted_lwr(x, y, xnew, f=1/3,iter=2,intercept=True):
  # we need decision trees
  # for training the boosted method we use x and y
  model1 = Lowess_AG_MD(f=f,iter=iter,intercept=intercept) # we need this for training the Decision Tree
  model1.fit(x,y)
  residuals1 = y - model1.predict(x)
  model2 = Lowess_AG_MD(f=f,iter=iter,intercept=intercept)
  #model2 = RandomForestRegressor(n_estimators=200,max_depth=9)
  model2.fit(x,residuals1)
  output = model1.predict(xnew) + model2.predict(xnew)
  return output 
 ```
 
 We will now change it slightly to take user-defined choices for each of the regressors for model1 and model2. 
 
 ```python
 def user_boosted_lwr(x, y, xnew, f=1/3, iter=2, intercept=True, model1_type="Lowess_AG_MD", model2_type="Lowess_AG_MD"):
    # Define the first model for training the decision tree
    if model1_type == "Lowess_AG_MD":
        model1 = Lowess_AG_MD(f=f, iter=iter, intercept=intercept)
    elif model1_type == "RandomForestRegressor":
        model1 = RandomForestRegressor(n_estimators=200,max_depth=5)
    else:
        raise ValueError("please choose 'Lowess_AG_MD' or 'RandomForestRegressor'.")
    model1.fit(x, y)
    residuals1 = y - model1.predict(x)
    
    # Define the second model for fitting the residuals
    if model2_type == "Lowess_AG_MD":
        model2 = Lowess_AG_MD(f=f, iter=iter, intercept=intercept)
    elif model2_type == "RandomForestRegressor":
        model2 = RandomForestRegressor(n_estimators=200,max_depth=5)
    else:
        raise ValueError("please choose 'Lowess_AG_MD' or 'RandomForestRegressor'.")

    model2.fit(x, residuals1)
    output = model1.predict(xnew) + model2.predict(xnew)
    return output
```

To demonstrate, we will look at it using the concrete dataset (using the distance function, kernels, and Lowess_AG_MD defined in class) :

```python
data = pd.read_csv('/content/drive/MyDrive/22 23 - Junior Yr/Adv Applied Machine Learning/1. Preliminaries, Intro to Locally Weighted Regression/concrete.csv')
data

x = data.loc[:,'cement':'age'].values
y = data['strength'].values

xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.3,shuffle=True,random_state=123)

yhat = user_boosted_lwr(xtrain, ytrain, xtest,f=25/len(xtrain),iter=1,intercept=True, model1_type="Lowess_AG_MD", model2_type="Lowess_AG_MD")

mse(ytest,yhat)
```
57.73151261936397

I will now try it with the RandomForestRegressor.

```python
yhat = user_boosted_lwr(xtrain, ytrain, xtest,f=25/len(xtrain),iter=1,intercept=True, model1_type="Lowess_AG_MD", model2_type="RandomForestRegressor")

mse(ytest,yhat)
```
42.90724190818058
