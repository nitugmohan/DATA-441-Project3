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
