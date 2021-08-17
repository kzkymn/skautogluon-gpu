# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from skautogluon import TabularPredictorWrapper
import autogluon.core as ag
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

# %%
# load datasets same as train.csv and test.csv in https://autogluon.s3.amazonaws.com/datasets/Inc/
X, y = sklearn.datasets.fetch_openml(data_id=179, return_X_y=True)

X_train, X_test_base, y_train, y_test_base = train_test_split(
    X, y, test_size=0.33, random_state=42)

X_val = X_test_base[:int(len(X_test_base)/2)]
y_val = y_test_base[:int(len(X_test_base)/2)]
X_test = X_test_base[int(len(X_test_base)/2):]
y_test = y_test_base[int(len(y_test_base)/2):]

# %%
# we specify eval-metric just for demo (unnecessary as it's the default)
metric = 'accuracy'

nn_options = {  # specifies non-default hyperparameter values for neural network models
    # number of training epochs (controls training time of NN models)
    'num_epochs': 10,
    # learning rate used in training (real-valued hyperparameter searched on log-scale)
    'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
    # activation function used in NN (categorical hyperparameter, default = first entry)
    'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),
    # each choice for categorical hyperparameter 'layers' corresponds to list of sizes for each NN layer to use
    'layers': ag.space.Categorical([100], [1000], [200, 100], [300, 200, 100]),
    # dropout probability (real-valued hyperparameter)
    'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),
}

gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
    # number of boosting rounds (controls training time of GBM models)
    'num_boost_round': 100,
    # number of leaves in trees (integer hyperparameter)
    'num_leaves': ag.space.Int(lower=26, upper=66, default=36),
}

hyperparameters = {  # hyperparameters of each model type
    'GBM': gbm_options,
    'NN': nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
}  # When these keys are missing from hyperparameters dict, no models of that type are trained

time_limit = 2*60  # train various models for ~2 min
num_trials = 5  # try at most 5 different hyperparameter configurations for each type of model
# to tune hyperparameters using Bayesian optimization routine with a local scheduler
search_strategy = 'auto'

hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
    'num_trials': num_trials,
    'scheduler': 'local',
    'searcher': search_strategy,
}

predictor = TabularPredictorWrapper(eval_metric=metric).fit(
    X=X_train, y=y_train, X_tuning=X_val, y_tuning=y_val,
    time_limit=time_limit,
    hyperparameters=hyperparameters,
    hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)

# %%
predictor = TabularPredictorWrapper(eval_metric=metric).fit(
    X=X_train, y=y_train, X_tuning=X_val, y_tuning=y_val,
    time_limit=time_limit)

# %%
y_pred = predictor.predict(X_test)
print("Predictions:  ", list(y_pred)[:5])
perf = predictor.evaluate(X_test, y_test, auxiliary_metrics=False)

# %%
results = predictor.fit_summary()

# %%
predictor = TabularPredictorWrapper(eval_metric=metric).fit(X=X_train,
                                                            y=y_train,
                                                            num_bag_folds=5, num_bag_sets=1, num_stack_levels=1,
                                                            # last  argument is just for quick demo here, omit it in real applications
                                                            hyperparameters={'NN': {'num_epochs': 2}, 'GBM': {
                                                                'num_boost_round': 20}},
                                                            )
# %%
# folder where to store trained models
save_path = 'agModels-predictOccupation'

predictor = TabularPredictorWrapper(eval_metric=metric, path=save_path).fit(
    X_train, y_train, auto_stack=True,
    # last 2 arguments are for quick demo, omit them in real applications
    time_limit=30, hyperparameters={'NN': {'num_epochs': 2}, 'GBM': {'num_boost_round': 20}}
)

# %%
# `predictor.path` is another way to get the relative path needed to later load predictor.
predictor = TabularPredictorWrapper.load(save_path)

# %%
# Note: .iloc[0] won't work because it returns pandas Series instead of DataFrame
datapoint = X_test.iloc[[0]]
print(datapoint)
predictor.predict(datapoint)

# %%
# returns a DataFrame that shows which probability corresponds to which class
predictor.predict_proba(datapoint)

# %%
predictor.get_model_best()

# %%
predictor.leaderboard(extra_info=True, silent=True)

# %%
predictor.leaderboard(X_test, y_test, extra_metrics=[
                      'accuracy', 'balanced_accuracy', 'log_loss'], silent=True)

# %%
i = 0  # index of model to use
model_to_use = predictor.get_model_names()[i]
model_pred = predictor.predict(datapoint, model=model_to_use)
print("Prediction from %s model: %s" % (model_to_use, model_pred.iloc[0]))

# %%
all_models = predictor.get_model_names()
model_to_use = all_models[i]
specific_model = predictor._predictor._trainer.load_model(model_to_use)

# Objects defined below are dicts of various information (not printed here as they are quite large):
model_info = specific_model.get_info()
predictor_information = predictor.info()

# %%
y_pred_proba = predictor.predict_proba(X_test)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_proba)

# %%
perf = predictor.evaluate(X_test, y_test)

# %%
predictor.feature_importance(X_test, y_test)

# %%
predictor.persist_models()

num_test = 20
preds = np.array(['']*num_test, dtype='object')
for i in range(num_test):
    datapoint = X_test.iloc[[i]]
    pred_numpy = predictor.predict(datapoint, as_pandas=False)
    preds[i] = pred_numpy[0]

perf = predictor.evaluate_predictions(
    y_test[:num_test], preds, auxiliary_metrics=True)
print("Predictions: ", preds)

# free memory by clearing models, future predict() calls will load models from disk
predictor.unpersist_models()

# %%
additional_ensembles = predictor.fit_weighted_ensemble(
    expand_pareto_frontier=True)
print("Alternative ensembles you can use for prediction:", additional_ensembles)

predictor.leaderboard(only_pareto_frontier=True, silent=True)

# %%
model_for_prediction = additional_ensembles[0]
predictions = predictor.predict(X_test, model=model_for_prediction)
# delete these extra models so they don't affect rest of tutorial
predictor.delete_models(models_to_delete=additional_ensembles, dry_run=False)

# %%
refit_model_map = predictor.refit_full()
print("Name of each refit-full model corresponding to a previous bagged ensemble:")
print(refit_model_map)
predictor.leaderboard(X_test, y_test, silent=True)

# %%
# specify much longer time limit in real applications
student_models = predictor.distill(time_limit=30)
print(student_models)
preds_student = predictor.predict(X_test, model=student_models[0])
print(f"predictions from {student_models[0]}:", list(preds_student)[:5])
predictor.leaderboard(X_test, y_test)

# %%
presets = ['good_quality_faster_inference_only_refit',
           'optimize_for_deployment']
predictor_light = TabularPredictorWrapper(eval_metric=metric).fit(
    X_train, y_train, presets=presets, time_limit=30)

# %%
predictor_light = TabularPredictorWrapper(eval_metric=metric).fit(
    X_train, y_train, hyperparameters='very_light', time_limit=30)

# %%
excluded_model_types = ['KNN', 'NN', 'custom']
predictor_light = TabularPredictorWrapper(eval_metric=metric).fit(
    X_train, y_train, excluded_model_types=excluded_model_types, time_limit=30)
