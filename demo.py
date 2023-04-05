import wandb
import random
from surprise import Dataset

from surprise import SVD,NormalPredictor
from surprise.model_selection import GridSearchCV
from surprise import Dataset, Reader

data = Dataset.load_builtin("ml-100k")

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-test-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "SVD",
    "dataset": "ml-100k",
    }
)

for n_epochs in [20, 30, 40, 50]: 
    param_grid = {'n_factors':[50],'n_epochs':[n_epochs],  'lr_all':[0.005],'reg_all':[0.02]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
    params = gs.best_params['rmse']
    best_rmse = gs.best_score['rmse']
    
    # log metrics to wandb 
    wandb.log({"n_factors": n_epochs, "rmse": best_rmse})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-test-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "SVD",
    "dataset": "ml-100k",
    }
)

for n_factors in [20, 30, 40, 50]: 
    param_grid = {'n_factors':[n_factors],'n_epochs':[50],  'lr_all':[0.005],'reg_all':[0.02]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
    params = gs.best_params['rmse']
    best_rmse = gs.best_score['rmse']
    
    # log metrics to wandb 
    wandb.log({"n_factors": n_factors, "rmse": best_rmse})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()