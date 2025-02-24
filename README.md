# Report: Scalability of Support Vector Machines
Dmitri Azih and Dylan Rivas

## Purposes

- Understanding the most popular Support Vector Machines (SVMs).
- Designing a basic algorithm to generate linearly separable datasets.
- Understanding and implementing the basic linear SVM with soft margin.
- Understanding and using the built-in SVMs in scikit-learn.
- Evaluating the scalability of SVMs on linearly separable datasets.

## Implementation Tasks

1. **LinearSVC Class**: 
   - Implement a Python class `LinearSVC` to learn a linear Support Vector Classifier (SVC) from a training dataset.
     - Constructor: Initializes SVC with learning rate, epochs, and random seed.
     - Training: Trains the SVC using soft-margin with hinge loss and L2-regularization.
     - Functions: Implement `net_input` for preactivation and `predict` for generating predictions.
2. **Data Generation**: 
   - Write a function `make_classification` to generate linearly separable data based on a random separation hyperplane.
     - The function should randomly generate a vector, sample points, and assign labels based on the hyperplane.
     - Split data into 70% training and 30% testing datasets.
3. **Scalability Investigation**: 
   - Investigate the scalability of the `LinearSVC` with different dataset sizes (e.g., `d = 10, 50, 100, 500, 1000` and `n = 500, 1000, 5000, 10000, 100000`).
     - Measure time cost differences for each dataset combination.
4. **Performance Comparison**: 
   - Investigate the performance of solving primal vs dual problems using scikit-learn's `LinearSVC`.
     - Compare training time and prediction accuracy on the test dataset for different dataset sizes.

