import numpy as np

from performance_metrics import calc_root_mean_squared_error


def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    ''' Perform n-fold cross validation for a specific sklearn estimator object

    Args
    ----
    estimator : any regressor object with sklearn-like API
        Supports 'fit' and 'predict' methods.
    x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
        Input measurements ("features") for all examples of interest.
        Each row is a feature vector for one example.
    y_N : 1D numpy array, shape (n_examples,)
        Output measurements ("responses") for all examples of interest.
        Each row is a scalar response for one example.
    n_folds : int
        Number of folds to divide provided dataset into.
    random_state : int or numpy.RandomState instance
        Allows reproducible random splits.

    Returns
    -------
    train_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for train set for fold f
    test_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for test set for fold f

    Examples
    --------
    # Create simple dataset of N examples where y given x
    # is perfectly explained by a linear regression model
    >>> N = 101
    >>> n_folds = 7
    >>> x_N3 = np.random.RandomState(0).rand(N, 3)
    >>> y_N = np.dot(x_N3, np.asarray([1., -2.0, 3.0])) - 1.3337
    >>> y_N.shape
    (101,)

    >>> import sklearn.linear_model
    >>> my_regr = sklearn.linear_model.LinearRegression()
    >>> tr_K, te_K = train_models_and_calc_scores_for_n_fold_cv(
    ...                 my_regr, x_N3, y_N, n_folds=n_folds, random_state=0)

    # Training error should be indistiguishable from zero
    >>> np.array2string(tr_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'

    # Testing error should be indistinguishable from zero
    >>> np.array2string(te_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'
    '''
    train_error_per_fold = np.zeros(n_folds, dtype=np.float32)
    test_error_per_fold = np.zeros(n_folds, dtype=np.float32)

    if isinstance(y_N, np.float64):
        train_fold_ids_list, test_fold_ids_list = make_train_and_test_row_ids_for_n_fold_cv(y_N, 
                                                n_folds, random_state=random_state)
    else:
        train_fold_ids_list, test_fold_ids_list = make_train_and_test_row_ids_for_n_fold_cv(len(y_N), 
                                                    n_folds, random_state=random_state)

    
    # TODO loop over folds and compute the train and test error
    # for the provided estimator
    
    for fold in range (n_folds):
        
        # Get the list of train ids / fold ids
        train_fold_ids = train_fold_ids_list[fold]
        test_fold_ids = test_fold_ids_list[fold]
        
        # Training and testing features/targets 
        x_train_NF = x_NF[train_fold_ids]
        y_train_N = y_N[train_fold_ids]
        x_test_NF = x_NF[test_fold_ids]
        y_test_N = y_N[test_fold_ids]
        
        # Fit the model based off of training data
        estimator.fit(x_train_NF, y_train_N)
        y_train_pdict = estimator.predict(x_train_NF)
        y_test_pdict = estimator.predict(x_test_NF)
        
        # calculate MSE to measure diff between measured/predicted values
        fold_train_error = calc_root_mean_squared_error(y_train_N, y_train_pdict)
        fold_test_error = calc_root_mean_squared_error(y_test_N, y_test_pdict)
        
        # append model error results 
        train_error_per_fold[fold] = fold_train_error
        test_error_per_fold[fold] = fold_test_error

    return train_error_per_fold, test_error_per_fold


def make_train_and_test_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
    ''' Divide row ids into train and test sets for n-fold cross validation.

    Will *shuffle* the row ids via a pseudorandom number generator before
    dividing into folds.

    Args
    ----
    n_examples : int
        Total number of examples to allocate into train/test sets
    n_folds : int
        Number of folds requested
    random_state : int or numpy RandomState object
        Pseudorandom number generator (or seed) for reproducibility

    Returns
    -------
    train_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N
    test_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N

    Guarantees for Return Values
    ----------------------------
    Across all folds, guarantee that no two folds put same object in test set.
    For each fold f, we need to guarantee:
    * The *union* of train_ids_per_fold[f] and test_ids_per_fold[f]
    is equal to [0, 1, ... N-1]
    * The *intersection* of the two is the empty set
    * The total size of train and test ids for any fold is equal to N

    Examples
    --------
    >>> N = 11
    >>> n_folds = 3
    >>> tr_ids_per_fold, te_ids_per_fold = (
    ...     make_train_and_test_row_ids_for_n_fold_cv(N, n_folds))
    >>> len(tr_ids_per_fold)
    3

    # Count of items in training sets
    >>> np.sort([len(tr) for tr in tr_ids_per_fold])
    array([7, 7, 8])

    # Count of items in the test sets
    >>> np.sort([len(te) for te in te_ids_per_fold])
    array([3, 4, 4])

    # Test ids should uniquely cover the interval [0, N)
    >>> np.sort(np.hstack([te_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    # Train ids should cover the interval [0, N) TWICE
    >>> np.sort(np.hstack([tr_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
            8,  9,  9, 10, 10])
    '''
    if hasattr(random_state, 'rand'):
        # Handle case where provided random_state is a random generator
        # (e.g. has methods rand() and randn())
        random_state = random_state # just remind us we use the passed-in value
    else:
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    # TODO obtain a shuffled order of the n_examples
    shuffled_ids = np.arange(n_examples)
    random_state.shuffle(shuffled_ids)
    
    # floor divide to get each fold size
    fold_size = n_examples // n_folds
    
    train_ids_per_fold = list()
    test_ids_per_fold = list()
    
    # TODO establish the row ids that belong to each fold's
    # train subset and test subset

    fold_sizes = []
    for fold in range(n_folds):
        if fold < n_examples % n_folds:
            fold_sizes.append(n_examples // n_folds + 1)
        else:
            fold_sizes.append(n_examples // n_folds)

    # establish start
    fold_start = 0

    for fold in range(n_folds):
        # establish end of fold
        # make sure fold_end doesn't exceed n_examples
        fold_end = min(fold_start + fold_sizes[fold], n_examples)

        # split shuffled ids into train/test sets
        test_fold_ids = shuffled_ids[fold_start:fold_end]
        train_fold_ids = np.concatenate([shuffled_ids[:fold_start], shuffled_ids[fold_end:]])
        
        # update fold start
        fold_start = fold_start + fold_sizes[fold]

        # add these train/test indeces sets parallel to each other
        train_ids_per_fold.append(train_fold_ids)
        test_ids_per_fold.append(test_fold_ids)        
    
    return train_ids_per_fold, test_ids_per_fold