import numpy as np
import torch
from .utils import *
from sklearn.base import BaseEstimator

from sklearn.model_selection import GridSearchCV
from sklearn.utils.extmath import fast_logdet
from sklearn.base import BaseEstimator
import numpy as np


class Random(object):
    
    def __init__(self):
        pass

    def fit(self, X):
        
        p = X[0].shape[-1]
        self.delta = 2*(torch.rand(p, p)-0.5)
        self.delta = self.delta + self.delta.T
        return self.delta


class Pinv(object):
    def __init__(self, mode='diff'):
        self.mode = mode

    def fit(self, X):
        """
        Pinv of cov to estimate precision matrix. 
        Differential network is inferred by substraction of two precision matrices.

        Args:
            X : X[0] and X[1] are array-like with shape (n,p). n is the number of samples, p is the dimension of variabels.

        """        
        if self.mode == 'diff':
            cov1 = np.corrcoef(X[0].T)
            cov2 = np.corrcoef(X[1].T)
            x1 = np.linalg.pinv(cov1, hermitian=True)
            x2 = np.linalg.pinv(cov2, hermitian=True)
            self.delta = theta2partial(x2) - theta2partial(x1)
        elif self.model == 'single':
            self.delta = np.linalg.pinv(X)
        return self.delta


class NetDiff(object):
    def __init__(self):
        import rpy2
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri

        rpy2.robjects.numpy2ri.activate()
        self.NetDiff = importr("NetDiff")
        pass

    def fit(self, X):
        """
        NetDiff from R.

        Args:
            X : X[0] and X[1] are array-like with shape (n,p). n is the number of samples, p is the dimension of variabels.

        """        
        # X shape: (2n, p)
        from sklearn import preprocessing
        X = np.concatenate((X[0],X[1]))
        X = preprocessing.scale(X)
        n = X.shape[0]
        partition = ["state1"] * int(n / 2) + ["state2"] * int(n / 2)
        partition = np.array(partition)
        results = self.NetDiff.netDiff(X, partition)
        self.theta1 = results[0][0]
        self.theta2 = results[0][1]
        self.delta = self.theta2 - self.theta1
        return self.delta


class BDGraph(object):
    
    def __init__(self, iter=5000):
        """_summary_

        Args:
            iter (int, optional): MCMC sampling number. Defaults to 5000.
        """        
        import rpy2
        self.iter = iter
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri

        rpy2.robjects.numpy2ri.activate()
        self.BDgraph = importr("BDgraph")
        pass

    def fit(self, X):
        """
        BDgraph from R.

        Args:
            X : X[0] and X[1] are array-like with shape (n,p). n is the number of samples, p is the dimension of variabels.

        """    
        X1 = X[0]
        X2 = X[1]
        
        results1 = self.BDgraph.bdgraph(X1, method="gcgm", iter=self.iter )
        results2 = self.BDgraph.bdgraph(X2, method="gcgm", iter=self.iter )

        self.theta1 = results1[1] 
        self.theta2 = results2[1] 
        self.delta = self.theta1 - self.theta2
        return self.delta




class JointGraphicalLasso(BaseEstimator):


    def __init__(self, lambda1=0.1, lambda2=0.1):
        """
        Parameters
        ----------
        lambda1 : float (must >=0)
            the parameter of sparsity penalty in JGL. The default is 0.1.
        lambda2 : float (must >=0)
            the parameter of similarity penalty in JGL. The default is 0.1.

        Returns
        -------
        None.
        """
        assert lambda1>=0, "lambda1 must >=0!"
        assert lambda2>=0, "lambda2 must >=0!"
        
        import rpy2
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri
        
        rpy2.robjects.numpy2ri.activate()
        self.JGL = importr('JGL')
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        assert (self.lambda1 >= 0 and self.lambda2 >= 0), "lambda1 and lambda2 should be > 0."

    def fit(self, X: np.ndarray, y=None):
        """
        
        JGL from R.

        Args:        

            X (np.ndarray): X[0] and X[1] are array-like with shape (n,p). n is the number of samples, p is the dimension of variabels.
            y (_type_, optional): Defaults to None.

        """        
        X = np.asarray(X)
        if X.shape[0] == 2:
            pass
        elif X.shape[1] == 2:
            X = np.transpose(X, (1,0,2))
        assert X.shape[0] == 2, "X shape should be (2,n,p)"
        X = [X_ for X_ in X]
        result = self.JGL.JGL(X, lambda1=self.lambda1,
                              lambda2=self.lambda2, return_whole_theta=True)
        self.precision1 = result[0][0]
        self.precision2 = result[0][1]
        self.delta = self.precision1 - self.precision2
        return self

    def get_params(self, deep=True):
        return {
            'lambda1': self.lambda1,
            'lambda2': self.lambda2
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y=None):
        X = np.asarray(X)
        X = np.asarray(X)
        if X.shape[0] == 2:
            pass
        elif X.shape[1] == 2:
            X = np.transpose(X, (1,0,2))
        cov1 = np.cov(X[0].T)
        cov2 = np.cov(X[1].T)
        precision1 = self.precision1
        precision2 = self.precision2
        p = precision1.shape[0]
        log_likelihood_ = - np.sum(cov1 * precision1) + fast_logdet(precision1)
        log_likelihood_ += - np.sum(cov2 * precision2) + fast_logdet(precision2)
        log_likelihood_ -= p * np.log(2 * np.pi)
        log_likelihood_ /= 2.

        return log_likelihood_


class JointGraphicalLassoCV(JointGraphicalLasso):
    """
    Parameters
    ----------
    grid_len : TYPE, optional
        Parameters grid length. The default is 5.
    verbose : TYPE, optional
        The larger, the more information output, 0 for no output, 3 for most output.
        The default is 0.
    n_refinement : TYPE, optional
        The numbers of refining the parameters grid. The default is 4.

    Returns
    -------
    None.

    """

    def __init__(self, grid_len=3, verbose=3, n_refinement=3):
        super(JointGraphicalLassoCV, self).__init__()
        self.grid_len = grid_len
        self.verbose = verbose
        self.n_refinement = n_refinement
        self.lambda1_max = 10
        self.lambda2_max = 10
        self.lambda1_min = 1e-3
        self.lambda2_min = 1e-3
        # JGL
        import rpy2
        import os
        # os.environ['R_HOME'] = "C:/PROGRA~1/R/R-3.5.1"
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        self.JGL = importr('JGL')

    def fit(self, X, y=None):
        """
        JGL cross validation using JGL.
        Args:

            X (np.ndarray): X[0] and X[1] are array-like with shape (n,p). n is the number of samples, p is the dimension of variabels.
            y (_type_, optional): Defaults to None.

        """        
        if self.verbose: print('Fitting JGL with CV...')
        X = np.asarray(X)
        X_ = np.transpose(X, (1,0,2))
        self.best_params = None
        for i in range(self.n_refinement):
            self.param_grid = {
                'lambda1': np.logspace(np.log10(self.lambda1_min),
                                       np.log10(self.lambda1_max), self.grid_len),
                'lambda2': np.logspace(np.log10(self.lambda2_min),
                                       np.log10(self.lambda2_max), self.grid_len),
            }
            self.cv = GridSearchCV(JointGraphicalLasso(),
                                   param_grid=self.param_grid, verbose=self.verbose)
            # n_samples must be first place. X_ : (n, 2, p)
            self.cv.fit(X_, y)
            self.index = self.cv.cv_results_['rank_test_score']
            if self.best_params == self.cv.best_params_:
                break
            self.best_params = self.cv.best_params_
            self.lambda1_min = self.best_params['lambda1']/2
            self.lambda1_max = self.best_params['lambda1']*2
            self.lambda2_min = self.best_params['lambda2']/2
            self.lambda2_max = self.best_params['lambda2']*2

        X = [X_ for X_ in X]
        result = self.JGL.JGL(X, lambda1=self.best_params['lambda1'],
                              lambda2=self.best_params['lambda2'], return_whole_theta=True)
        self.precision1 = result[0][0]
        self.precision2 = result[0][1]
        self.delta = self.precision1 - self.precision2

        return self