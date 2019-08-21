import os
import pprint as pp

import multiprocessing

from surprise import Reader
from surprise import Dataset

from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV

from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import NormalPredictor
from surprise import SlopeOne
from surprise import CoClustering

from surprise.model_selection import cross_validate

print('SETTING DATA:')

#obtaining cores for the running machine
cores = multiprocessing.cpu_count()

# path of dataset file
file_path = os.path.expanduser('./dataset/ratings.csv')

print("Loading Dataset...")
reader = Reader(line_format='user item rating', sep=',', rating_scale=[0.5, 5], skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)
print("Done.")
print()

print()
print("Performing splits...")
kf = KFold(n_splits=5, random_state=0)
print("Done.")
print()
###############################################################

#NORMAL PREDICTOR

current_algo = NormalPredictor()
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True)
print()
###############################################################

#BASELINE ONLY

baseline_predictor_options = {
    'method': "sgd",
    'learning_rate': 0.005,
    'n_epochs': 50,  #
    #
    'reg': 0.02,  
}

current_algo = BaselineOnly(bsl_options=baseline_predictor_options, verbose=True)
cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True)
print()
###############################################################

#KNN BASIC

MAXIMUM_number_of_neighbors_to_consider = 40 
min_number_of_neighbors_to_consider = 5 


similarity_options = {
    'user_based': False, 
    'name': "pearson_baseline", 
    'min_support': 3,
    
}

current_algo = KNNBasic(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                        sim_options=similarity_options, verbose=True)

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True)

###############################################################

#KNN WITH MEANS

MAXIMUM_number_of_neighbors_to_consider = 35  
min_number_of_neighbors_to_consider = 5  


similarity_options = {
    'user_based': False,  
    'name': "pearson_baseline", 
    'min_support': 3,
    
}

current_algo = KNNWithMeans(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                        sim_options=similarity_options, verbose=True)

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True)

###############################################################

# KNN WITH Z SCORE

MAXIMUM_number_of_neighbors_to_consider = 35
min_number_of_neighbors_to_consider = 3  


similarity_options = {
    'user_based': False, 
    'name': "pearson_baseline", 
    'min_support': 3,
    
}

current_algo = KNNWithZScore(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                        sim_options=similarity_options, verbose=True)

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True)

###############################################################

#KNN BASELINE


similarity_options = {
    'user_based': False,  
    'name': "pearson_baseline", 
    'min_support': 3,

}

baseline_predictor_options = {
    'method': "sgd",
    'learning_rate': 0.0005,
    'n_epochs': 50,
    #
    'reg': 0.02, 
}

MAXIMUM_number_of_neighbors_to_consider2 = 40 
min_number_of_neighbors_to_consider2 = 5

current_algo = KNNBaseline(k=MAXIMUM_number_of_neighbors_to_consider, min_k=min_number_of_neighbors_to_consider,
                        sim_options=similarity_options, bsl_options=baseline_predictor_options, verbose=True)

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True)
print()

###############################################################

# SVD

current_algo = SVD(n_factors=100, n_epochs=40, biased=True, lr_all=.01, reg_all=.1, verbose=True)

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=cores)
print()

###############################################################

#SVD ++

current_algo = SVDpp(n_factors=100, n_epochs=15, lr_all=.01, reg_all=.1)

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=cores)
print()
###############################################################

#NMF
current_algo = NMF(n_factors=15,
    n_epochs=50,
    biased=False,
    reg_pu=0.06,
    reg_qi=0.06,
    reg_bu=0.02,
    reg_bi=0.02,
    lr_bu=0.005,
    lr_bi=0.005,
    init_low=0,
    init_high=1,
    random_state=None,
    verbose=False)

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True, n_jobs=cores)
print()

###############################################################

# SLOPE ONE

current_algo = SlopeOne()

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True)
print()

###############################################################

# CO-Clustering

coclustering_predictor_options = {
    'n_cltr_u': 10,
    'n_cltr_i': 4,  
    'n_epochs': 50, 
    'verbose': True,
}

current_algo = CoClustering(n_cltr_u=10, n_cltr_i=4, n_epochs=50, verbose=True)

cross_validate(current_algo, data, measures=['RMSE'], cv=kf, verbose=True)
print()

###############################################################
###################################
###
### Hyperparameter Optimization 
###
###################################
#KNN BASELINE
grid_of_parameters = {
    'k': [35, 40, 45],
    'min_k': [5,6,7], 
    'sim_options': { 
        'user_based': [False, True],
        'name': ['pearson_baseline'], 
        'min_support': [2,3], 
    }
}



gsKNNBaseline = GridSearchCV(KNNBaseline,
                  param_grid = grid_of_parameters,
                  measures=['rmse'],
                  cv=kf, n_jobs=cores,
                  joblib_verbose=10)

gsKNNBaseline.fit(data)
print()


print()
print("BEST_SCORE: " + str(gsKNNBaseline.best_score['rmse']))


print()
print("BEST_PARAMETERS: ")
pp.pprint(gsKNNBaseline.best_params['rmse'])
print()

print()

###############################################################

#SVD

grid_of_parameters = {'n_factors': [90, 100],
                      'biased' : [True],
                      'n_epochs': [45,50,55],
                      'lr_all' : [.01,.02,.03],
                      'reg_all':[.1],
                     'verbose':[True]}

gsSVD = GridSearchCV(SVD,
                  param_grid = grid_of_parameters,
                  measures=['rmse'],
                  cv=kf, n_jobs=cores)
gsSVD.fit(data)


print("BEST_SCORE: " + str(gsSVD.best_score['rmse']))

print("BEST_PARAMETERS: ")
pp.pprint(gsSVD.best_params['rmse'])

