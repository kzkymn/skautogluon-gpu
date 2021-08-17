# The purpose of this code is to show that autogluon-wrapper can
# do the same thing as the original AutoGluon can.
# Therefore, much of this code is almost same as that of AutoGluon Quickstart.
# %%
from skautogluon import TabularPredictorWrapper
import sklearn.datasets
from sklearn.model_selection import train_test_split

# %%
# Reading the data that is equivalent to train.csv in AutoGluon Quickstart.
X, y = sklearn.datasets.fetch_openml(data_id=179, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# %%
# Unlike the original TabularPredictor, it is not necessary to explicitly pass
# the label name of the target variable as the constructor argument of TabularPredictorWrapper.
save_path = 'agModels-predictClass'
predictor = TabularPredictorWrapper(path=save_path).fit(X_train, y_train)

# %%
results = predictor.fit_summary()

# %%
# From TabularPredictorWrapper, You can refer to some properties the original TabularPredictor has.
print("AutoGluon infers problem type is: ", predictor.problem_type)
print("AutoGluon identified the following types of features:")
print(predictor.feature_metadata)

# %%
# unnecessary, just demonstrates how to load previously-trained predictor from file
predictor = TabularPredictorWrapper.load(save_path)

y_pred = predictor.predict(X_test)
print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(
    y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

# %%
predictor.leaderboard(X_test, y_test, silent=True)

# %%
predictor.predict(X_test, model='LightGBM')

# %%
time_limit = 60
metric = 'roc_auc'
predictor = TabularPredictorWrapper(eval_metric=metric)
predictor.fit(X_train, y_train, time_limit=time_limit, presets='best_quality')
predictor.leaderboard(X_test, y_test, silent=True)

# %%
