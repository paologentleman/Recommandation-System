# Recommandation-System

![alt text](https://miro.medium.com/max/1000/1*kjouN-zV6BgpmCl5SnEjGQ.jpeg)

The goal of this project is to improve the performance of a recommendation-system by using non-trivial algorithms and by performing hyper-parameters tuning.

Using data relative to movie ratings of users, we apply all algorithms for recommendation made available by [Surprise](http://surpriselib.com/) libraries, according to their default configuration. Then we improve the quality of both **KNNBaseline** and **SVD** methods, by performing hyper-parameters tuning over five-folds.

____

# Files

* **RecommandationSys.py**: python file which contains the core code and implementation of the Recommandation System and the hyperparameter CrossValidation.

* **ratings.csv**: Csv file containing the ratings for each user of all the movies in the catalogue.
