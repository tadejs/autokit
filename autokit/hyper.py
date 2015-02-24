import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, BaggingClassifier, BaggingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import Nystroem, RBFSampler

from autokit import models

import hyperopt

import autokit.hpsklearn as hpsklearn


class HyperML:
    def __init__(self, info, verbose = True, debug_mode = False, max_cycles = 100):
        self.label_num=info['label_num']
        self.target_num=info['target_num']
        self.task = info['task']
        self.metric = info['metric']
        self.is_multilabel = info['task'] == 'multilabel.classification'
        self.is_regression = info['task'] == 'regression'
        
        self.postprocessor = None #models.MultiLabelEnsemble(LogisticRegression(), balance=False) # To calibrate proba

        if self.is_regression:
            if info['is_sparse'] is True:
                classifier = hpsklearn.any_sparse_regressor('mod')
            else:
                classifier = hpsklearn.any_regressor('mod')   
        else:
            #if self.is_multilabel:
            #    classifier = hpsklearn.random_forest('mod.random_forest', n_jobs=-1)
            if info['is_sparse'] is True:
                classifier = hpsklearn.any_sparse_classifier('mod')
            else:
                classifier = hpsklearn.any_classifier('mod')   
    
        self.model = hpsklearn.HyperoptEstimator(
            preprocessing=hpsklearn.components.any_preprocessing('pp', info['feat_num'], info['is_sparse']),
            classifier=classifier,
            algo=hyperopt.tpe.suggest,
            trial_timeout=info['time_budget'],# / max(1, info['label_num']), # seconds
            max_evals=max_cycles
        )
        
        if self.is_multilabel:
            self.model = models.MultiLabelEnsemble(self.model)
        
        if info['task']=='regression':
            self.predict_method = self.model.predict
        else:
            self.predict_method = self.model.predict_proba
    
    def start_fit(self, X, Y):
        self.iterator = self.model.fit_iter(X, Y)
        self.iterator.next() 
    
    def should_continue(self):
        return len(self.model.trials.trials) < self.model.max_evals
    
    def fit(self, X, Y):
        self.iterator.send(1) # -- try one more model     
        print(self.model.best_model())      
        if any(map(lambda x: not x, self.model.best_model())):
            raise Exception("no model trained")
        
        self.model.retrain_best_model_on_full_data(X, Y)            

        # Train a calibration model postprocessor
        if self.task != 'regression' and self.postprocessor!=None:
            Yhat = self.predict_method(X)
            self.postprocessor.fit(Yhat, Y)
        return self
        
    def predict(self, X):
        prediction = self.predict_method(X)
        # Calibrate proba
        if self.task != 'regression' and self.postprocessor!=None:          
            prediction = self.postprocessor.predict_proba(prediction)
        # Keep only 2nd column because the second one is 1-first    
        if self.target_num==1 and len(prediction.shape)>1 and prediction.shape[1]>1:
            prediction = prediction[:,1]
        # Make sure the normalization is correct
        if self.task=='multiclass.classification':
            eps = 1e-15
            norma = np.sum(prediction, axis=1)
            for k in range(prediction.shape[0]):
                prediction[k,:] /= sp.maximum(norma[k], eps)  
        return prediction
    
    def __repr__(self):
        return "HyperML : " + self.name

    def __str__(self):
        return "HyperML : \n" + str(self.model) 

class KaAutoML:
        
    def __init__(self, info, verbose=True, debug_mode=False):
        self.label_num=info['label_num']
        self.target_num=info['target_num']
        self.task = info['task']
        self.metric = info['metric']
        self.postprocessor = None
        #self.postprocessor = MultiLabelEnsemble(LogisticRegression(), balance=True) # To calibrate proba
        self.postprocessor = models.MultiLabelEnsemble(LogisticRegression(), balance=False) # To calibrate proba

        if info['task']=='regression':
            if info['is_sparse']==True:
                self.name = "BaggingRidgeRegressor"
                self.model = BaggingRegressor(base_estimator=Ridge(), n_estimators=1, verbose=verbose) # unfortunately, no warm start...
            else:
                self.name = "GradientBoostingRegressor"
                self.model = GradientBoostingRegressor(n_estimators=1, verbose=verbose, warm_start = True)
        else:
            if info['has_categorical']: # Out of lazziness, we do not convert categorical variables...
                self.name = "RandomForestClassifier"
                self.model = RandomForestClassifier(n_estimators=1, verbose=verbose) # unfortunately, no warm start...
            elif info['is_sparse']:                
                self.name = "BaggingNBClassifier"
                self.model = BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=1, verbose=verbose) # unfortunately, no warm start...                          
            else:
                self.name = "GradientBoostingClassifier"
                self.model = eval(self.name + "(n_estimators=1, verbose=" + str(verbose) + ", min_samples_split=10, random_state=1, warm_start = True)")
            if info['task']=='multilabel.classification':
                self.model = models.MultiLabelEnsemble(self.model)  
                          
        self.pipeline = Pipeline([('prep', FeatureUnion([
                                        ('nys', Nystroem(kernel = 'rbf', n_components = 100)), 
                                        ('std', StandardScaler(with_mean = not info['is_sparse']))
                                        ])),
                                  ('mod', self.model)])
        
        if info['task']=='regression':
            self.predict_method = self.pipeline.predict
        else:
            self.predict_method = self.pipeline.predict_proba
        
    def __repr__(self):
        return "TadejAutoML : " + self.name

    def __str__(self):
        return "TadejAutoML : \n" + str(self.pipeline) 

    def fit(self, X, Y):
        self.pipeline.fit(X,Y)
        # Train a calibration model postprocessor
        if self.task != 'regression' and self.postprocessor!=None:
            Yhat = self.predict_method(X)
            self.postprocessor.fit(Yhat, Y)
        return self
        
    def predict(self, X):
        prediction = self.predict_method(X)
        # Calibrate proba
        if self.task != 'regression' and self.postprocessor!=None:          
            prediction = self.postprocessor.predict_proba(prediction)
        # Keep only 2nd column because the second one is 1-first    
        if self.target_num==1 and len(prediction.shape)>1 and prediction.shape[1]>1:
            prediction = prediction[:,1]
        # Make sure the normalization is correct
        if self.task=='multiclass.classification':
            eps = 1e-15
            norma = np.sum(prediction, axis=1)
            for k in range(prediction.shape[0]):
                prediction[k,:] /= sp.maximum(norma[k], eps)  
        return prediction

