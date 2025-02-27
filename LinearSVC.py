import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#####################
# SVC Class

class LinearSVC:
    """Linear Support Vector Classifier (SVC) with Soft Margin.
    
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Number of passes over the training dataset.
    random_state : int
        Random number generator seed for random weight initialization.
    lambda_reg : float
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
    def __init__(self, eta=0.0005, n_iter=100, random_state=1, L2_reg=0.001):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.L2_reg = L2_reg
        
    def fit(self, X, y):
        """Fit training data using stochastic gradient descent."""
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0
        self.losses_ = []
        
        for epoch in range(self.n_iter):
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



###############################
#  Plotting Methods

def plot_2D(X, y, a, u):
    plt.figure(figsize=(8, 6))

    # Plot the hyperplane 
    x_hyperplane = np.linspace(-u, u, 100)
    y_hyperplane = (-a[0] * x_hyperplane) / a[1]
    plt.plot(x_hyperplane, y_hyperplane, color='black', label='Hyperplane')

    if y_pred is not None:
        # Plot given predicted data points 
        plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], color='green', marker='x', label='Class 1 (Predicted)', alpha=0.6)
        plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], color='orange', marker='x', label='Class -1 (Predicted)', alpha=0.6)
    else:
        # Plot all data points
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1 (True)', alpha=0.6)
        plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label='Class -1 (True)', alpha=0.6)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linearly Separable Data with Hyperplane and Data/Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3D(X, y, a, u, predictions=False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if predictions:
        # Plot given predicted data points 
        ax.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], X[y_pred == 1, 2], color='green', marker='x', label='Class 1 (Predicted)', alpha=0.6)
        ax.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], X[y_pred == -1, 2], color='orange', marker='x', label='Class -1 (Predicted)', alpha=0.6)
    else:
        # Plot all data points
        ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='blue', label='Class 1', alpha=0.6)
        ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='red', label='Class -1', alpha=0.6)

    # Plot the hyperplane (3D)
    xx, yy = np.meshgrid(np.linspace(-u, u, 10), np.linspace(-u, u, 10))
    zz = (-a[0] * xx - a[1] * yy) / a[2]
    ax.plot_surface(xx, yy, zz, color='black', alpha=0.5)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Linearly Separable Data in 3D')
    ax.legend() 
    plt.show()


###############################
#  Data Generation Method

def make_classification(d=10, n=500, u=1e6, random_seed=123, graph = False):
    """ Generates a set of linearly separable data 
        based on a random seperation hyperplane 
    """

    rgen = np.random.RandomState(random_seed)

    # Hyperplane: Randomly generate a d-dimensional vector a in range [-u, u]
    a = rgen.uniform(-u, u, d)

    # X: Randomly select n samples of values between [-u,u] in the d-th dimension
    X = rgen.uniform(-u, u, size=(n, d))

    # y: Label each xi such that if  ̄a^T*x < 0 then yi = −1, otherwise yi = 1
    y = np.sign(np.dot(X, a))

    # Subdivide the dataset to a training dataset (70%) and a test dataset (30%) with unique seed
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed+1)

    if graph and d in [2, 3]:
        if d == 2: plot_2D(X, y, a, u)
        else: plot_3D(X, y, a , u)

    return X, y, a
###################################################################

random_seed = 123

###############################
#  Example Data Creation/Training/Results/Plotting

example_results = False
if example_results:
    # Randomly generate a dth-dimension linearly separable data set with labels and seperating hyperplane
    d = 3
    n = 500
    u = 1e6
    X, y, a = make_classification(d=d, n=n, u=u, random_seed=random_seed, graph = True)

    # Subdivide the dataset to a training dataset (70%) and a test dataset (30%) with unique seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed+1)

    # Create and fit a SVC under training data
    svc = LinearSVC(eta=0.001, n_iter=50, random_state=random_seed, L2_reg=0.001)
    trained_svc = svc.fit(X_train, y_train)

    # Test model on test data
    y_pred = trained_svc.predict(X_test)

    # Calculate SVC's performance
    misclassifcations = np.sum(y_pred != y_test)
    accuracy = (((y_test.shape[0])-misclassifcations)/y_test.shape[0])*100
    print(f"Test data missclassifcations: {misclassifcations}\nAccuracy: {accuracy:.2f}")

    # Plot models predictions against hyperplane
    if d==2: plot_2D(X_test, y_pred, a, u)
    elif d==3: plot_3D(X_test, y_pred, a, u)


###############################
#  Explore Scalability

scalability = True
if scalability:

    u = 1e12
    d = [10, 50, 100, 500, 1000]
    n = [500, 1000, 5000, 10000, 100000]


    for ni in n:
        print(f"\nNumber of data points (n = {ni}):")
        for di in d:
            # Create linearly separable data
            X, y, a = make_classification(d=di, n=ni, u=u, random_seed=random_seed, graph = True)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed+1)

            # Create and fit a SVC under training data
            svc = LinearSVC(eta=0.0005, n_iter=100, random_state=random_seed, L2_reg=0.001)
            trained_svc = svc.fit(X_train, y_train)

            y_pred = trained_svc.predict(X_test)

            # Calculate SVC's performance
            misclassifcations = np.sum(y_pred != y_test)
            accuracy = (((y_test.shape[0])-misclassifcations)/y_test.shape[0])*100
            print(f"    Missclassifcations (d = {di}): {misclassifcations}\n    Accuracy: {accuracy:.2f}")
            
