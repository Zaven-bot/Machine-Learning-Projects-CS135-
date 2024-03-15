## HW2: Evaluating Binary Classifiers and Implementing Logistic Regression <br>

**Coding Tasks:** <br>
> **Problem 1: Binary Classifier for Cancer-Risk Screening**
>> * Implement calc_TP_TN_FP_FN in binary_metrics.py to calculate TP, TN, FP, and FN based on the provided true labels and predicted labels.
>> * Implement calc_ACC in binary_metrics.py to calculate the accuracy of the classifier, which is the ratio of correct predictions (TP + TN) to the total number of predictions.
>> * Implement calc_TPR in binary_metrics.py to calculate the true positive rate (TPR), also known as sensitivity or recall.
>> * Implement calc_PPV in binary_metrics.py to calculates the positive predictive value (PPV), also known as precision
>> * Implement performance metrics for binary predictions combining the implemented functions to compute and report various performance metrics such as accuracy, TPR, and PPV.

> **Problem 2: Computing the Loss for Logistic Regression without Numerical Issues**
>> * Implement calc_mean_binary_cross_entropy_from_probas in proba_metrics.py to compute the mean binary cross-entropy loss from true binary labels and predicted probabilities, ensuring numerical stability by preprocessing the probabilities.
>> * Implement my_logsumexp in logsumexp.py to compute the logarithm of the sum of exponentials of input values, ensuring numerical stability by using the logsumexp trick.
>> * Implement calc_mean_binary_cross_entropy_from_scores in proba_metrics.py to compute the mean binary cross-entropy loss from true binary labels and predicted scores.

**Report Tasks:**
>> * Analyze the accuracy of the "predict-0-always" classifier on the test set.
>> * Compare the ROC curves of the 2-feature and 3-feature models.
>> * Report the number of unnecessary biopsies saved by each thresholding strategy.
>> * Determine the best thresholding strategy for the screening task and the fraction of current biopsies that might be avoided.
>> * Plot ROC curves for both models.
>> * Summarize test-set performance of logistic regression models across different thresholds.
