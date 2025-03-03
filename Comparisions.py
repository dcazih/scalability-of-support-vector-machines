import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearSVC import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC as SklearnLinearSVC

###############################
#  Plotting Methods

def plot_2D(X, y, a, u, classtype="", title="Linearly Separable 2D Data with Hyperplane", ):
    plt.figure(figsize=(8, 6))

    # Plot the hyperplane 
    x_hyperplane = np.linspace(-u, u, 100)
    y_hyperplane = (-a[0] * x_hyperplane) / a[1]
    plt.plot(x_hyperplane, y_hyperplane, color='black', label='Hyperplane')

    # Plot all data points
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label=f'Class 1 {classtype}', alpha=0.6)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label=f'Class -1 {classtype}', alpha=0.6)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3D(X, y, a, u, classtype="", title="Linearly Separable #D Data with Hyperplane"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all data points
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], color='blue', label=f'Class 1 {classtype}', alpha=0.6)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], color='red', label=f'Class -1 {classtype}', alpha=0.6)

    # Plot the hyperplane (3D)
    xx, yy = np.meshgrid(np.linspace(-u, u, 10), np.linspace(-u, u, 10))
    zz = (-a[0] * xx - a[1] * yy) / a[2]
    ax.plot_surface(xx, yy, zz, color='black', alpha=0.5)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title(title)
    ax.legend() 
    plt.show()

def plot_df(df):
    fig, ax = plt.subplots(figsize=(10, 6)) 
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colLoc='center')
    plt.show()

    df.to_excel('performance_analysis.xlsx', index=False, engine='openpyxl')
    print(df)

###############################
#  Data Generation Method

def make_classification(d=10, n=500, u=1e6, random_seed=123):
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

    return X, y, a
###################################################################



###############################
#  Example Data Creation/Training/Results/Plotting

random_seed = 123

example_results = False
if example_results:
    # Randomly generate a dth-dimension linearly separable data set with labels and seperating hyperplane
    d = 2
    n = 100
    u = 1e6
    X, y, a = make_classification(d=d, n=n, u=u, random_seed=random_seed)

    # Graph distribution
    plot_2D(X, y, a, u, "(true)", "Demo of 2D Random Point Distribution for d = 2, n = 100")

    # Subdivide the dataset to a training dataset (70%) and a test dataset (30%) with unique seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed+1)

    # Create and fit a SVC under training data
    svc = LinearSVC(eta=0.001, epochs=50, random_state=random_seed, L2_reg=0.001)
    trained_svc = svc.fit(X_train, y_train)

    # Test model on test data
    y_pred = trained_svc.predict(X_test)

    # Calculate SVC's performance
    misclassifcations = np.sum(y_pred != y_test)
    accuracy = (((y_test.shape[0])-misclassifcations)/y_test.shape[0])*100
    print(f"Test data missclassifcations: {misclassifcations}\nAccuracy: {accuracy:.2f}")



###############################
#  Testing: Exploring Scalability

results = []

scalability = True
if scalability:

    u = 1e12
    d = [10, 50, 100, 500, 1000]
    n = [500, 1000, 5000, 10000, 100000]

    for ni in n:
        print(f"\nNumber of data points (n = {ni}):")
        for di in d:
            # Create linearly separable data
            X, y, a = make_classification(d=di, n=ni, u=u, random_seed=random_seed)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed+1)

            # Create and fit a SVC under training data
            svc = LinearSVC(eta=0.0005, epochs=100, random_state=random_seed, L2_reg=0.001)

            # GET the fitting time
            start_time = time.time()
            trained_svc = svc.fit(X_train, y_train)
            fit_time = time.time() - start_time

            # Calculate SVC's performance
            y_pred = trained_svc.predict(X_test)
            misclassifications = np.sum(y_pred != y_test)
            accuracy = (((y_test.shape[0]) - misclassifications) / y_test.shape[0]) * 100

            # Store the results for each combination
            results.append([ni, di, round(fit_time, 2), misclassifications, round(accuracy, 2)])

    # Tabulate performance
    df = pd.DataFrame(results, columns=["Data Points (n)", "Dimensions (d)", "Fit Time (s)", "Misclassifications", "Accuracy (%)"])
    plot_df(df)


###############################
#  Testing: Compare performance of LinearSVC implemenation with Scikit-Learn's LinearSVC

comparison = False
if comparison:
    print(f"\nComparison of LinearSVC's")
    # Initialize parameters for testing
    u = 1e12
    d = [10, 50, 100, 500, 1000]
    n = [500, 1000, 5000, 10000, 100000]

    # Loop through different dataset sizes
    for ni in n:
        print(f"\nNumber of data points (n = {ni}):")
        for di in d:
            print(f"\nNumber of feature dimensions (d = {di})")

            # Generate test data
            X, y, a = make_classification(d=di, n=ni, u=u, random_seed=random_seed)
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed+1)

            # Test the custom SVC from this script
            customSVC = LinearSVC(eta=0.0005, epochs=100, random_state=random_seed, L2_reg=-.001)
            # Time the training
            startTime = time.time()
            customSVC.fit(X_train, y_train)
            customTrainTime = time.time() - startTime
            # Get the accuracy
            customPredict = customSVC.predict(X_test)
            customAccuracy = accuracy_score(y_test, customPredict)

            # Test the sklearn SVC for primal problem
            skLearnSVC_primal = SklearnLinearSVC(dual=False, max_iter=1000, random_state=random_seed)
            # Time the training
            startTime = time.time()
            skLearnSVC_primal.fit(X_train, y_train)
            skLearnSVC_primalTrainTime = time.time() - startTime
            # Get the accuracy
            skLearnSVC_primalPredict = skLearnSVC_primal.predict(X_test)
            skLearnSVC_primalAccuracy = accuracy_score(y_test, skLearnSVC_primalPredict)

            # Test the sklearn SVC for dual problem
            skLearnSVC_dual = SklearnLinearSVC(dual=True, max_iter=1000, random_state=random_seed)
            # Time the training
            startTime = time.time()
            skLearnSVC_dual.fit(X_train, y_train)
            skLearnSVC_dualTrainTime = time.time() - startTime
            # Get the accuracy
            skLearnSVC_dualPredict = skLearnSVC_primal.predict(X_test)
            skLearnSVC_dualAccuracy = accuracy_score(y_test, skLearnSVC_primalPredict)

            # Print the results
            print(f"Custom SVC - Train Time: {customTrainTime:.4f}s, Accuracy: {customAccuracy:.4f}")
            print(f"Sklearn Primal SVC - Train Time: {skLearnSVC_primalTrainTime:.4f}s, Accuracy: {skLearnSVC_primalAccuracy:.4f}")
            print(f"Sklearn Dual SVC - Train Time: {skLearnSVC_dualTrainTime:.4f}s, Accuracy: {skLearnSVC_dualAccuracy:.4f}")

