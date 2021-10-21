# RecipeBandit
Implementation of k-armed bandit with Recipe/Review data from Food.com.

First download zip of data from kaggle link:
https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=interactions_test.csv

Unzip this and place all data into data directory /data


## Current State of Progress:

### data:

contains multiple csv files holding preprocessed and raw data regarding the recipes and interaction data (reviews)

### EDAS:

- RecipeEDA looks into the distributions of ratings, basic statistics regarding the data
- DimensionalityReduction looks into the SVD of a user ratings matrix, establishing methods to be used to reduce dimensionality of our sparse matrices
    - It also contains a previous implemented version of the environment from the tutorial
    
### BaselineRecipeBandit

This notebook includes all work done towards producing the first baseline recipe bandit.

A contextual multi-armed bandit in our context will be define as such:

a system that contains an agent with k-arms, where k is the number of recipes in our problem. Each round, the agent is given a global context vector, containing information relevant to the user for which the recipe is to be recommended, and a per arm matrix, containing sample_size vectors representing each recipe which the agents can possibly recommend. 

One of the simplest contextual bandits is the LinUCB algorithm, which maintains a model which takes both the user vector and recipe vector and performs ridge regression to predict an upper confidence bound for the expected reward for a given user and recipe. This is the simplest implementation in the tf-agents library and thus makes sense to use as the baseline as we move towards deep reinforcement learning.

For the baseline model, each user will be represented by a vector which is the size of the number of recipes in the problem where each value represents the rating the user gives the recipe. Each recipe vector is this matrix transpose such that a recipe vector contains all the user ratings for it.

The data being used is the train_interactions.csv, read into a matrix of dimensions # of users by # of recipes, which is then decomposed to the appropriate dimensions on the fly.

**NOTE**: SVD is used on these features maintaining dimensions which retain some threshold of explained variance of the data. SVD is appropriate since our ratings are sparse. 

The environment includes the required methods, observe and apply action, which basically fetches the data for context for a given run or provides the reward for the previous action. It does this by querying our decomposed matrices and ratings matrix.


### CONCERNS

- Bias from unreviewed recipes, in training and in evaluation. We are making an assumption that if a user doesn't review a recipe, they give it a 0 out of 5.

Very similar to the tf-agents example but somethings differ:
- Cannot use pure numpy/scipy SVD because evaluation with unseen recipe data (reviews from users that were missing) cannot be decomposed (sklearn provides a fit_transform function). 


#### ISSUES WITH TRAINING

Getting a full training loop has been difficult. Have adjusted the following parameters to reduce complexity:
- explained variance threshold for number of components taken from SVD, higher obviously means more features which means slower
- the number of recipes the bandit can suggest each round: the algorithm doesn't necessarily consider all recipes each iteration, instead they sample some recipes and recommend froma  smaller sample, learning more iteratively the correct weights for regression. This perhaps could be alleviated if obseervation could be passed by reference since passing the full matrix everytime is very computationally heavy.
- Number of training loops: number of times we update the bandit

Currently I am trying to reduce complexity to at least implement an evaluator.

**BIG COMPUTATIONAL BOOST:** The agent is the LinUCB agent, have done a pretty deep dive and found per arm configuration creates a single model vs the non per arm config (only user global context) creates a model for each arm (much heavier)
