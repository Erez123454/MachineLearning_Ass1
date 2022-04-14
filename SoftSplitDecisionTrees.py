from random import random
import numpy as np
from sklearn import tree
from sklearn.tree import _tree
from sklearn.utils.validation import check_is_fitted, _deprecate_positional_args


class SoftSplitDecisionTreeClassifier(tree.DecisionTreeClassifier):
    def __init__(self, *,n=100,
                 alphaProbability=0.1,
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
        self.n=n
        self.alphaProbability=alphaProbability


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
        proba = self.predict_proba(X,check_input)
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

    def predictWrapper(self,row,predictionFunction):
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
        probabilisticPredictionFunction = self._generateProbabilisticPredictionFunction(self.alphaProbability)
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

    def _generateProbabilisticPredictionFunction(self, alphaProbability):
        '''
        Private util function to generate probabilistic prediction function.
        :param alphaProbability:
        :type alphaProbability:
        :return: function which accept single record
        :rtype: function
        '''
        tree = self.tree_

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
            if tree.feature[currNode] != _tree.TREE_UNDEFINED:
                randomProb = random()
                sampleValue = sample[tree.feature[currNode]]
                if sampleValue <= tree.threshold[currNode]:
                    if (1 - alphaProbability) >= randomProb:
                        return _traverse(sample, tree.children_left[currNode])
                    else:
                        return _traverse(sample, tree.children_right[currNode])
                else:
                    if (1 - alphaProbability) >= randomProb:
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


class SoftSplitDecisionTreeRegressor(tree.DecisionTreeRegressor):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"mse", "friedman_mse", "mae", "poisson"}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion and minimizes the L2 loss
        using the mean of each terminal node, "friedman_mse", which uses mean
        squared error with Friedman's improvement score for potential splits,
        "mae" for the mean absolute error, which minimizes the L1 loss using
        the median of each terminal node, and "poisson" which uses reduction in
        Poisson deviance to find splits.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 0.24
            Poisson deviance criterion.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    min_impurity_split : float, default=0
        Threshold for early stopping in tree growth. A node will split
        if its impurity is above the threshold, otherwise it is a leaf.

        .. deprecated:: 0.19
           ``min_impurity_split`` has been deprecated in favor of
           ``min_impurity_decrease`` in 0.19. The default value of
           ``min_impurity_split`` has changed from 1e-7 to 0 in 0.23 and it
           will be removed in 1.0 (renaming of 0.25).
           Use ``min_impurity_decrease`` instead.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details.

        .. versionadded:: 0.22

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
    """

    def __init__(self, *,n=100,
                 alphaProbability=0.1,
                 criterion="mse",
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
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)
        self.n=n
        self.alphaProbability=alphaProbability

    def predict(self, X, check_input=True):
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

            alphaProbability : double, default=0 probability to choose the opposite decision path.

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
        probabilisticPredictionFunction = self._generateProbabilisticPredictionFunction(self.alphaProbability)
        probaList = [self.predictWrapper(row,probabilisticPredictionFunction) for row in X]
        proba = np.concatenate(probaList, axis=0)

        # Regression
        if self.n_outputs_ == 1:
            return proba[:, 0]

        else:
            return proba[:, :, 0]

    def _generateProbabilisticPredictionFunction(self, alphaProbability):
        '''
        Private util function to generate probabilistic prediction function.
        :param alphaProbability:
        :type alphaProbability:
        :return: function which accept single record
        :rtype: function
        '''
        tree = self.tree_

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
            if tree.feature[currNode] != _tree.TREE_UNDEFINED:
                randomProb = random()
                sampleValue = sample[tree.feature[currNode]]
                if sampleValue <= tree.threshold[currNode]:
                    if (1 - alphaProbability) >= randomProb:
                        return _traverse(sample, tree.children_left[currNode])
                    else:
                        return _traverse(sample, tree.children_right[currNode])
                else:
                    if (1 - alphaProbability) >= randomProb:
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
