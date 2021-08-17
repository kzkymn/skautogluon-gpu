from copy import copy
import os
from typing import Dict, List, Optional, Union

from autogluon.tabular import TabularPredictor
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

SKAUTOGLUONDIR = 'sk_autogluon'
SKAUTOGLUONFILE = 'sk_autogluon.joblib'


class TabularPredictorWrapper(BaseEstimator):
    def __init__(self,
                 label='y',
                 problem_type=None,
                 eval_metric=None,
                 path=None,
                 verbosity=2,
                 sample_weight=None,
                 weight_evaluation=False,
                 **kwargs
                 ):
        self._candidated_y_label = label
        self._problem_type = problem_type
        self._eval_metric = eval_metric
        self._path = path
        self.verbosity = verbosity
        self.sample_weight = sample_weight
        self.weight_evaluation = weight_evaluation
        self._init_kwargs = kwargs
        self._are_fit_options_set = False
        self._feature_metadata = None

    @property
    def class_labels(self):
        return self._predictor._learner.class_labels

    @property
    def class_labels_internal(self):
        return self._predictor._learner.label_cleaner.ordered_class_labels_transformed

    @property
    def class_labels_internal_map(self):
        return self._predictor._learner.label_cleaner.inv_map

    @property
    def quantile_levels(self):
        return self._predictor._learner.quantile_levels

    @property
    def eval_metric(self):
        try:
            return self._predictor._learner.eval_metric
        except NameError:
            return self._eval_metric

    @property
    def problem_type(self):
        try:
            return self._predictor._learner.problem_type
        except NameError:
            return self._problem_type

    @property
    def feature_metadata(self):
        try:
            return self._predictor._trainer.feature_metadata
        except NameError:
            return self._feature_metadata

    @property
    def feature_metadata_in(self):
        return self._predictor._learner.feature_generator.feature_metadata_in

    @property
    def path(self):
        try:
            return self._predictor._learner.path
        except NameError:
            return self._path

    def __create_y_label(self, X):
        X_col_set = set(X.columns)
        if self._candidated_y_label in X_col_set:
            for i in range(10):
                new_candidated_y_label = self._candidated_y_label + \
                    '_' + str(i)
                if new_candidated_y_label not in X_col_set:
                    return new_candidated_y_label
            raise ValueError(
                f'fit() tried to use \'{self._candidated_label}\' as the name of the objective variable but failed. Because X has columns which names are same or very similar as \'{self._candidated_label}\'. Please set an alternative label name explicitly at __init__() of this class.')
        return self._candidated_y_label

    def __create_data(self, X, y):
        if y is None:
            raise ValueError(
                'If you set the value of tuning_X, you must set the value of tuning_y.')
        data = X.copy()
        data[self.label] = y
        return data

    def fit(self, X, y,
            X_tuning=None,
            y_tuning=None,
            time_limit=None,
            presets: Optional[Union[List, str, Dict]] = None,
            hyperparameters=None,
            feature_metadata='infer',
            **kwargs):
        self.label = self.__create_y_label(X)
        self._predictor = TabularPredictor(self.label,
                                           problem_type=self._problem_type,
                                           eval_metric=self._eval_metric,
                                           path=self._path,
                                           verbosity=self.verbosity,
                                           sample_weight=self.sample_weight,
                                           weight_evaluation=self.weight_evaluation,
                                           **self._init_kwargs)

        train_data = self.__create_data(X, y)
        tuning_data = None

        if X_tuning is not None:
            tuning_data = self.__create_data(X_tuning, y_tuning)

        if presets is None:
            self._predictor.fit(train_data,
                                tuning_data=tuning_data,
                                time_limit=time_limit,
                                hyperparameters=hyperparameters,
                                feature_metadata=feature_metadata,
                                **kwargs)
        elif isinstance(presets, (list, str, dict)):
            self._predictor.fit(train_data,
                                tuning_data=tuning_data,
                                time_limit=time_limit,
                                presets=presets,
                                hyperparameters=hyperparameters,
                                feature_metadata=feature_metadata,
                                **kwargs)
        else:
            raise ValueError('Type of presets must be list str or dict.')

        self._is_fitted_ = True
        self.__save_sk_autogluon_object()

        return self

    def fit_extra(self, hyperparameters, time_limit=None, base_model_names=None, **kwargs):
        check_is_fitted(self)
        self._predictor.fit_extra(hyperparameters=hyperparameters,
                                  time_limit=time_limit,
                                  base_model_names=base_model_names,
                                  **kwargs)
        return self

    def predict(self, data, model=None, as_pandas=True):
        check_is_fitted(self)
        return self._predictor.predict(data, model=model, as_pandas=as_pandas)

    def predict_proba(self, data, model=None, as_pandas=True, as_multiclass=True):
        check_is_fitted(self)
        return self._predictor.predict_proba(data, model=model, as_pandas=as_pandas, as_multiclass=as_multiclass)

    def evaluate(self, X, y,
                 model=None, silent=False, auxiliary_metrics=True, detailed_report=False) -> dict:
        check_is_fitted(self)
        data = self.__create_data(X, y)
        return self._predictor.evaluate(data, model=model, silent=silent,
                                        auxiliary_metrics=auxiliary_metrics,
                                        detailed_report=detailed_report)

    def evaluate_predictions(self, y_true, y_pred, silent=False, auxiliary_metrics=True, detailed_report=False) -> dict:
        check_is_fitted(self)
        return self._predictor.evaluate_predictions(y_true, y_pred, silent=silent,
                                                    auxiliary_metrics=auxiliary_metrics,
                                                    detailed_report=detailed_report)

    def leaderboard(self, X=None, y=None, extra_info=False, extra_metrics=None, only_pareto_frontier=False, silent=False):
        check_is_fitted(self)
        data = None
        if X is not None:
            data = self.__create_data(X, y)
        return self._predictor.leaderboard(data=data, extra_info=extra_info,
                                           extra_metrics=extra_metrics,
                                           only_pareto_frontier=only_pareto_frontier,
                                           silent=silent)

    def fit_summary(self, verbosity=3):
        check_is_fitted(self)
        return self._predictor.fit_summary(verbosity=verbosity)

    def transform_features(self, data=None, model=None, base_models=None, return_original_features=True):
        return self._predictor.transform_features(data=data,
                                                  model=model,
                                                  base_models=base_models,
                                                  return_original_features=return_original_features)

    def transform_labels(self, labels, inverse=False, proba=False):
        return self._predictor.transform_labels(labels, inverse=inverse, proba=proba)

    def feature_importance(self, X=None, y=None, model=None, features=None, feature_stage='original', subsample_size=1000, time_limit=None, num_shuffle_sets=None, include_confidence_band=True, silent=False):
        check_is_fitted(self)
        data = self.__create_data(X, y)
        return self._predictor.feature_importance(data=data, model=model, features=features, feature_stage=feature_stage,
                                                  subsample_size=subsample_size, time_limit=time_limit,
                                                  num_shuffle_sets=num_shuffle_sets,
                                                  include_confidence_band=include_confidence_band, silent=silent)

    def persist_models(self, models='best', with_ancestors=True, max_memory=0.1) -> list:
        return self._predictor.persist_models(models=models, with_ancestors=with_ancestors, max_memory=max_memory)

    def unpersist_models(self, models='all'):
        return self._predictor.unpersist_models(models=models)

    def refit_full(self, model='all'):
        check_is_fitted(self)
        return self._predictor.refit_full(model=model)

    def get_model_best(self):
        check_is_fitted(self)
        return self._predictor.get_model_best()

    def get_model_full_dict(self):
        check_is_fitted(self)
        return self._predictor.get_model_full_dict()

    def info(self):
        return self._predictor.info()

    def fit_weighted_ensemble(self, base_models: list = None, name_suffix='Best', expand_pareto_frontier=False, time_limit=None):
        return self._predictor.fit_weighted_ensemble(base_models=base_models,
                                                     name_suffix=name_suffix,
                                                     expand_pareto_frontier=expand_pareto_frontier,
                                                     time_limit=time_limit)

    def get_oof_pred(self, model: str = None, transformed=False, train_data=None, internal_oof=False) -> 'pd.Series':
        check_is_fitted(self)
        return self._predictor.get_oof_pred(model=model, transformed=transformed, train_data=train_data, internal_oof=internal_oof)

    def get_oof_pred_proba(self, model: str = None, transformed=False, as_multiclass=True, train_data=None, internal_oof=False) -> Union[pd.DataFrame, pd.Series]:
        check_is_fitted(self)
        return self._predictor.get_oof_pred_proba(model=model, transformed=transformed,
                                                  as_multiclass=as_multiclass, train_data=train_data,
                                                  internal_oof=internal_oof)

    @ property
    def positive_class(self):
        return self._predictor.positive_class

    def load_data_internal(self, data='train', return_X=True, return_y=True):
        return self._predictor.load_data_internal(data=data, return_X=return_X, return_y=return_y)

    def save_space(self, remove_data=True, remove_fit_stack=True, requires_save=True, reduce_children=False):
        self._predictor.save_space(remove_data=remove_data, remove_fit_stack=remove_fit_stack,
                                   requires_save=requires_save, reduce_children=reduce_children)

    def delete_models(self, models_to_keep=None, models_to_delete=None, allow_delete_cascade=False, delete_from_disk=True, dry_run=True):
        check_is_fitted(self)
        self._predictor.delete_models(models_to_keep=models_to_keep, models_to_delete=models_to_delete,
                                      allow_delete_cascade=allow_delete_cascade, delete_from_disk=delete_from_disk,
                                      dry_run=dry_run)

    def get_model_names(self, stack_name=None, level=None, can_infer: bool = None, models: list = None) -> list:
        check_is_fitted(self)
        return self._predictor.get_model_names(stack_name=stack_name, level=level, can_infer=can_infer, models=models)

    def get_model_names_persisted(self) -> list:
        return self._predictor.get_model_names_persisted()

    def distill(self, X=None, y=None, tuning_X=None, tuning_y=None, augmentation_X=None, time_limit=None, hyperparameters=None, holdout_frac=None,
                teacher_preds='soft', augment_method='spunge', augment_args={'size_factor': 5, 'max_size': int(1e5)}, models_name_suffix=None, verbosity=None):

        train_data = None
        tuning_data = None

        if X is not None:
            train_data = self.__create_data(X, y)
        if tuning_X is not None:
            tuning_data = self.__create_data(tuning_X, tuning_y)

        return self._predictor.distill(train_data=train_data, tuning_data=tuning_data, augmentation_data=augmentation_X,
                                       time_limit=time_limit, hyperparameters=hyperparameters, holdout_frac=holdout_frac,
                                       teacher_preds=teacher_preds, augment_method=augment_method, augment_args=augment_args,
                                       models_name_suffix=models_name_suffix, verbosity=verbosity)

    def plot_ensemble_model(self, prune_unused_nodes=True) -> str:
        check_is_fitted(self)
        return self._predictor.plot_ensemble_model(prune_unused_nodes=prune_unused_nodes)

    def __save_sk_autogluon_object(self):
        sk_autogluon_dump_dir = os.path.join(
            self._predictor.path, SKAUTOGLUONDIR)
        os.makedirs(sk_autogluon_dump_dir, exist_ok=True)
        sk_autogluon_file_path = os.path.join(
            sk_autogluon_dump_dir, SKAUTOGLUONFILE)
        copy_self = copy(self)
        copy_self._predictor = None
        joblib.dump(copy_self, sk_autogluon_file_path)

    def save(self):
        self._predictor.save()
        self.__save_sk_autogluon_object()

    @classmethod
    def load(cls, path: str, verbosity: int = None) -> 'TabularPredictor':
        sk_autogluon_file_path = os.path.join(
            path, SKAUTOGLUONDIR, SKAUTOGLUONFILE)
        tabular_predictor_wrapper = joblib.load(sk_autogluon_file_path)
        _predictor = TabularPredictor.load(path=path, verbosity=verbosity)
        tabular_predictor_wrapper._predictor = _predictor
        return tabular_predictor_wrapper
