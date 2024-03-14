import numpy as np
import sklearn
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.neural_network

from activations import (
    sigmoid, softmax, relu, identity)


def predict_0_hidden_layer(x_NF, w_arr, b_arr, output_activation):
    """ Make predictions for a neural net with no hidden layers

    This function demonstrates 3 special cases for an MLP with 0 hidden layers
    1. identity activation : equivalent to linear regression
    2. sigmoid activation : equivalent to binary logistic regression
    3. softmax activation : equivalent to multi-class logistic regression

    Args
    ----
    x_NF : 2D numpy array, shape (N, F) = (n_samples, n_features)
        Input features

    w_arr : 1D or 2D numpy array, shape (n_features, n_outputs)
        For single output, this may be a 1D array of shape (n_features,)

    b_arr : 1D numpy array, shape (n_outputs,)
        For single output, this may be a scalar float

    output_activation : callable
        Activation function for the output layer.
        Given an input array, must return output array of same shape

    Returns
    -------
    yhat_NC : 1D or 2D numpy array:
        shape (N,C) = (n_samples, n_outputs) if n_outputs > 1, else shape (N,C) = (n_samples,)
        Predicted values using the specified neural network configuration
        
        Suppose we had N=3 examples, F=1 features, and n_outputs = 1
        * if output_activation == identity, return array of real values
            e.g., input: [x1, x2, x3] --> output:[2.5, -6.7, 12]
        * if output_activation == sigmoid, return an array of probabilities
            e.g., input: [x1, x2, x3] --> output:[0.3, 0.8, 1.0]

        Suppose we had N=2 examples, F=1 features, and n_outputs = 3
        * if output_activation == softmax, return an array of proba vectors.
            e.g., input: [x1, x2] --> output:[[0.2, 0.4, 0.4], [0.8, 0.2, 0.]]

    Examples
    --------
    --------------------------------------
    Test Cases for predict_0_hidden_layers
    --------------------------------------

    1. linear regression model

    >>> x_NF, y_N = sklearn.datasets.make_regression(n_samples=100, n_features=5, noise=1, random_state=42)
    >>> reg = sklearn.linear_model.LinearRegression().fit(x_NF, y_N)
    >>> w = reg.coef_
    >>> print(w.shape)
    (5,)
    >>> b = reg.intercept_
    >>> round(b, 3)
    -0.009
    >>> yhat_N = predict_0_hidden_layer(x_NF, w, b, output_activation=identity)
    >>> print(yhat_N.shape)
    (100,)
    >>> np.allclose(yhat_N, reg.predict(x_NF))
    True

    2. binary classifier via logistic regression

    >>> x_NF, y_N = sklearn.datasets.make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    >>> clf = sklearn.linear_model.LogisticRegression().fit(x_NF, y_N)
    >>> w = np.squeeze(clf.coef_)
    >>> print(w.shape)
    (5,)
    >>> b = clf.intercept_[0]
    >>> print(round(b, 3))
    0.078
    >>> yproba1_N = predict_0_hidden_layer(x_NF, w, b, output_activation=sigmoid)
    >>> print(yproba1_N.shape)
    (100,)
    >>> np.allclose(yproba1_N, clf.predict_proba(x_NF)[:,1])
    True

    3. multi-class logistic regression model.

    >>> x_NF, y_N = sklearn.datasets.make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=4, random_state=42)
    >>> multi_clf = sklearn.linear_model.LogisticRegression(multi_class="multinomial").fit(x_NF, y_N)
    >>> w = multi_clf.coef_.T
    >>> print(w.shape) #(n_features, n_classes)
    (5, 4)
    >>> b = multi_clf.intercept_
    >>> print(b.shape) #(n_classes,)
    (4,)
    >>> yproba_NC = predict_0_hidden_layer(x_NF, w, b, output_activation=softmax)
    >>> yproba_NC.shape
    (100, 4)
    >>> np.allclose(yproba_NC, multi_clf.predict_proba(x_NF))
    True


    """
    linear_trans = np.dot(x_NF, w_arr) + b_arr
    pred = output_activation(linear_trans)
    return pred


def predict_n_hidden_layer(
        x_NF, w_list, b_list,
        hidden_activation=relu, output_activation=softmax):
    """ Make predictions for an MLP with zero or more hidden layers

    Parameters:
    -----------
    x_NF : numpy array of shape (n_samples, n_features)
        Input data for prediction.

    w_list : list of numpy array, length is n_layers
        Each entry represents 2D weight array for corresponding layer
        Shape of each entry is (n_inputs, n_outputs)
        Layers are ordered from input to output in predictive order

    b_list : list of numpy array, length is n_layers
        Each entry represents the intercept aka bias array for a specific layer
        Shape of each entry is (n_outputs,)
        Layers are ordered from input to output in predictive order

    hidden_activation : callable, optional (default=relu)
        Activation function for all hidden layers.

    output_activation : callable, optional (default=softmax)
        Activation function for the output layer.

    Returns:
    --------
    yhat_NC : 1D or 2D numpy array:
        shape (N,C) = (n_samples, n_outputs) if n_outputs > 1, else shape (N,C) = (n_samples,)
        Predicted values (for regression) or probabilities (if classification)
        Each row corresponds to corresponding row of x_NF input array.

        Suppose we had N=2 examples, F=1 features, and n_outputs = 1
        * if output_activation == sigmoid, return an array of proba vectors of label 1.
            e.g., input: [x1, x2] --> output:[[0.2], [0.8]]

        Suppose we had N=2 examples, F=1 features, and n_outputs = 3
        * if output_activation == softmax, return an array of proba vectors.
            e.g., input: [x1, x2] --> output:[[0.2, 0.4, 0.4], [0.8, 0.2, 0.]]

    Examples
    _______
    1. predict probabilities for all classes

    >>> x_NF, y_N = sklearn.datasets.make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=4, random_state=42)
    >>> mlp_2hidden = sklearn.neural_network.MLPClassifier(\
        hidden_layer_sizes=[2],activation='relu', solver='lbfgs', random_state=1)
    >>> mlp_2hidden = mlp_2hidden.fit(x_NF, y_N)
    >>> yproba_N2 = predict_n_hidden_layer(x_NF, mlp_2hidden.coefs_, mlp_2hidden.intercepts_)
    >>> np.round(yproba_N2[:2], 2)
    array([[0.85, 0.05, 0.1 , 0.  ],
        [0.97, 0.02, 0.02, 0.  ]])
    >>> print(np.sum(yproba_N2[:2], axis=1))
    [1. 1.]
    >>> ideal_yproba_N2 = mlp_2hidden.predict_proba(x_NF)
    >>> np.allclose(yproba_N2, ideal_yproba_N2)
    True
    >>> np.round(ideal_yproba_N2[:2], 2)
    array([[0.85, 0.05, 0.1 , 0.  ],
        [0.97, 0.02, 0.02, 0.  ]])

    2. Try replacing the softmax (default) with identity

    >>> yhat_N2 = predict_n_hidden_layer(x_NF,
    ... 	mlp_2hidden.coefs_, mlp_2hidden.intercepts_, output_activation=identity)
    >>> np.round(yhat_N2[:2], 2)
    array([[  3.18,   0.42,   0.99, -10.06],
        [  4.82,   0.76,   0.67, -14.12]])

    """
    n_layers = len(w_list)
    assert n_layers == len(b_list)

    # Forward propagation: start from the input layer
    out_arr = x_NF

    for layer_id in range(n_layers):
        # Get w and b arrays for current layer
        w_arr = w_list[layer_id]
        b_arr = b_list[layer_id]

        # Perform the linear operation: X · w + b
        out_arr = np.dot(out_arr, w_arr) + b_arr

        # Perform the non-linear activation of current layer
        if layer_id < n_layers - 1:
            out_arr = hidden_activation(out_arr)
        else:
            out_arr = output_activation(out_arr)

    out_arr = np.squeeze(out_arr) # reduce unnecessary dimension for single output

    return out_arr
