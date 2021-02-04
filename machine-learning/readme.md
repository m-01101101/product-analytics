# Machine learning

Notes taken from a range of courses and compiled here. 

Courses includes:

- Supervised learning with `scikit learn` on [DataCamp](https://learn.datacamp.com/courses/supervised-learning-with-scikit-learn)

## What is machine learning

A programme can make a decision using data rather than being explicitly programmed. For example, is this email spam or not, or clustering wikipedia articles into categories.

When they are labels presents this is "supervised learning", i.e. spam or not spam. "Unsupervised learning" does not use labels. It is essentially the task of uncovering hidden patterns and structure from unlabelled data.

In "reinforcement learning" the software interacts with an environment. The agents then "figure out" how to optimise their behaviour given a system of rewards and punishments. Deep learning is a form of reinforcement learning.

## Supervised learning

We have `features` (predictor or independent variables) and `target` (response or dependent) variable.

We'll present the data in a table format, with each column representing a variable and each row an instance.

Predictor variables are either;

- discrete: they consist of categories or booleans, this is a classification task
- continuous: they consist of a continuous distribution of numbers, this is a regression task

### k-Nearest neighbours (KNN)

> The only things the algorithm requires are:
>
> - Some notion of distance
> - An assumption that points that are close to one another are similar
> 
> _Grus, Joel. Data Science from Scratch_

Classifying the blue point:

<img src="md_refs/knn1.png>

The algorithm is creating a set of "decision boundaries", this is intuitive when viewed in 2D:

<img src="md_refs/knn2.png>

As the boundary gets smoother, the model is less complex, less suspectible to "noise" and should be less overfitted. This typically means a higher `k`;

<img src="md_refs/knn3.png>

__Accuracy__ = correction predictions / total data points (using unseen data only!!)
