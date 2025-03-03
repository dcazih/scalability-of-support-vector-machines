import numpy as np

#####################
# SVC Class

class LinearSVC:
    """Linear Support Vector Classifier (SVC) with Soft Margin.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Number of passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    L2_reg : float
        Regularization parameter for L2 regularization.
    
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
        Hinge loss function values in each epoch.
    """
    def __init__(self, eta=0.0005, epochs=100, random_state=1, L2_reg=0.001):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state
        self.L2_reg = L2_reg
        
    def fit(self, X, y):
        """Fit training data using stochastic gradient descent."""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0
        self.losses_ = []
        
        for epoch in range(self.epochs):
            loss = 0
            for xi, yi in zip(X, y):
                margin = yi * self.net_input(xi)
                if margin < 1:
                    self.w_ -= self.eta * (self.L2_reg * self.w_ - yi * xi)
                    self.b_ += self.eta * yi
                    loss += 1 - margin
                else:
                    self.w_ -= self.eta * self.L2_reg * self.w_
            self.losses_.append(loss / len(y))
        
        return self

    # Pre-activation function
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Predict class labels using sign function."""
        return np.where(self.net_input(X) >= 0, 1, -1)

###################################################################
