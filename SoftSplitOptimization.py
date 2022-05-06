import math
from random import random
import numpy as np
from sklearn import tree
from sklearn.tree import _tree
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args


class SoftSplitOptimizationDecisionTreeClassifier(tree.DecisionTreeClassifier):
    def __init__(self, *, n=100,alphaProbability=0.1, nearSensitivity=10000,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 ccp_alpha=0.0):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)
        self.n = n
        self.alphaProbability=alphaProbability
        self.nearSensitivity=nearSensitivity

    def predict(self, X, check_input=True):
        # check_is_fitted(self)
        # X = self._validate_X_predict(X, check_input)
        # return self.predictWrapper(X)
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        proba = self.predict_proba(X, check_input)
        n_samples = X.shape[0]

        # Classification
        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            class_type = self.classes_[0].dtype
            predictions = np.zeros((n_samples, self.n_outputs_),
                                   dtype=class_type)
            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(
                    np.argmax(proba[:, k], axis=1),
                    axis=0)

            return predictions

    def predictWrapper(self, row, predictionFunction):
        '''
        Function to wrap the single prediction process, the function will
        run the prediction for n iteration
        :param X:
        :type X:
        :param n:
        :type n:
        :param alphaProbability:
        :type alphaProbability:
        :return:
        :rtype:
        '''
        predictions = np.array([predictionFunction(row) for i in range(self.n)])
        avgPredictions = np.array([np.array([val for val in i]) for i in predictions.mean(0)])
        return avgPredictions

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.

        The predicted class probability is the fraction of samples of the same
        class in a leaf.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        alphaProbability : double, default=0 probability to choose the opposite decision path.

        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
            such arrays if n_outputs > 1
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        probabilisticPredictionFunction = self._generateProbabilisticPredictionFunction()
        probaList = [self.predictWrapper(row, probabilisticPredictionFunction) for row in X]
        proba = np.concatenate(probaList, axis=0)

        if self.n_outputs_ == 1:
            proba = proba[:, :self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, :self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba

    def _generateProbabilisticPredictionFunction(self):
        '''
        Private util function to generate probabilistic prediction function.
        :return: function which accept single record
        :rtype: function
        '''
        tree = self.tree_
        nearSensitivity = self.nearSensitivity
        alphaProbability = self.alphaProbability

        def _traverse(sample, currNode):
            '''
            private function to traverse in the decision tree
            :param sample: single record without the target variable
            :type sample: pandas dataframe
            :param currNode: index of the current node in the tree struct
            :type currNode: int
            :return: array of n-output of the classes, hold values for the record
            :rtype: nd.array
            '''

            def __calcSigmod(x):
                # return max(0.5,( 1 / (1 + np.exp(-(1/threshold)*(x**2)))) -0.1)
                return max(0.5,( 1 / (1 + np.exp(-nearSensitivity*(x**2)))) - alphaProbability)

            if tree.feature[currNode] != _tree.TREE_UNDEFINED:
                randomProb = random()
                sampleValue = sample[tree.feature[currNode]]
                distance = sampleValue - tree.threshold[currNode]
                prob = __calcSigmod(distance)
                if sampleValue <= tree.threshold[currNode]:
                    if randomProb <= prob:
                        return _traverse(sample, tree.children_left[currNode])
                    else:
                        return _traverse(sample, tree.children_right[currNode])
                else:
                    if randomProb <= prob:
                        return _traverse(sample, tree.children_right[currNode])
                    else:
                        return _traverse(sample, tree.children_left[currNode])
            else:
                value = tree.value[currNode]
                return value

        def _predict_single(sample):
            '''
            Util function to wrap the sample and the tree traverse for lazy use
            :param sample:
            :type sample:
            :return: application of the traverse with the given sample and the root
            :rtype:
            '''
            return _traverse(sample, 0)

        return _predict_single
