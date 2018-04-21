'''
    Perform Support Vector Classification on the given non-linearly separable 2D data.
    
    @author: JustaGist
    @package: ml_playground

    @Usage: Run python svc_nonlinear_data.py -h to see usage.

'''


import sys
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
import argparse
import numpy as np
from sklearn.svm import SVC

np.random.seed(0)

def create_nonlinearly_separable_2d_data(num_pts = 200, show_data = True):
    '''
        Function for creating a set of random non-linearly separable dataset of two dimension.

        Returns X: datapoints (2D)
                Y: corresponding labels (1 or -1)

    '''

    X_xor = np.random.randn(num_pts, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                           X_xor[:, 1] > 0)

    y_xor = np.where(y_xor, 1, -1)

    if show_data:

        plt.scatter(X_xor[y_xor == 1, 0],
                    X_xor[y_xor == 1, 1],
                    c='b', marker='x',
                    label='1')

        plt.scatter(X_xor[y_xor == -1, 0],
                    X_xor[y_xor == -1, 1],
                    c='r',
                    marker='s',
                    label='-1')

        plt.title("Dataset")
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    return X_xor, y_xor


class SupportVectorClassifier:

    ''' 
        A Classifier that can classify a 2D dataset of points based on Support Vector Machines.

    '''

    def __init__(self, datapoints, labels, kernel_type = 'linear', **kwargs):

        self.x_ = datapoints
        self.y_ = labels

        self._initialise_svc(kernel_type, **kwargs)



    def _initialise_svc(self, kernel_type, **kwargs):
        '''
            Create the svc kernel. 

            @args: kernel_type: The kernel type to use to create the SVM: 'linear' or 'rbf'
                 : C          : The penalty for misclassifying a data point
                 : gamma      : 'Spread' of the rbf kernel
        '''

        def _create_linear_kernel_svc(C = 1, random_state = 0, **kwargs):

            self.svm_ = SVC(kernel=kernel_type, C = C, random_state = random_state)

        def _create_rbf_kernel_svc(C = 1, gamma = 0.01, random_state = 0):

            self.svm_ = SVC(kernel=kernel_type, random_state = 0, gamma = gamma, C = C)


        if kernel_type == 'linear':

            _create_linear_kernel_svc(**kwargs)

        elif kernel_type == 'rbf':

            _create_rbf_kernel_svc(**kwargs)

        else:
            raise Exception("Unrecognized kernel type Error")



    def fit(self, show_plot = True):
        '''
            Fit the dataset using the SVM kernel model. This will separate the datapoints into two across the kernel.

        '''

        self.svm_.fit(self.x_, self.y_)

        if show_plot:

                self._plot_decision_regions(self.x_, self.y_)

                plt.legend(loc='upper left')
                plt.tight_layout()
                plt.show()



    def _plot_decision_regions(self, X, y, test_idx=None, resolution=0.02):
        '''
            Function for plotting the separation created by the SVC.

        ''' 

        # ----- setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # ----- plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))

        Z = self.svm_.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)

        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):

            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

        # ----- highlight test samples
        if test_idx:
            # ----- plot all samples
            X_test, y_test = X[test_idx, :], y[test_idx]

            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='',
                        alpha=1.0,
                        linewidths=1,
                        marker='o',
                        s=55, label='test set')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Support Vector Classifier Implementation to test the effect of C and gamma parameters on the performance using linear and rbf kernels.')
    parser.add_argument("-K", "--kernel_type", help="the kernel type to use to create the SVM classifier: linear/rbf", type=str)
    parser.add_argument("-C", default = 1, help="C is a parameter of the SVC learner and is the penalty for misclassifying a data point. When C is small, the classifier is okay with misclassified data points (high bias, low variance). When C is large, the classifier is heavily penalized for misclassified data and therefore bends over backwards avoid any misclassified data points (low bias, high variance).", type=float)
    parser.add_argument("-g", "--gamma", default = 0.01, help="gamma is a parameter of the RBF kernel and can be thought of as the 'spread' of the kernel and therefore the decision region. When gamma is low, the 'curve' of the decision boundary is very low and thus the decision region is very broad. When gamma is high, the 'curve' of the decision boundary is high, which creates islands of decision-boundaries around data points. ",type=float)
    parser.add_argument("-N","--num_pts", default=200, help="Number of random datapoints to create", type=int)
    parser.add_argument("-v", "--visualize", default = 1, help = "Whether to show the initial datapoints or not (0/1)", type=int)

    args = parser.parse_args()

    if args.kernel_type is None:

        print "Invalid usage. Specify Kernel type. Use -h flag to see usage."
        sys.exit()

    data, labels = create_nonlinearly_separable_2d_data(num_pts = args.num_pts, show_data = bool(args.visualize))
    
    SupportVectorClassifier(datapoints = data, labels = labels, kernel_type = args.kernel_type, random_state = 0, C = args.C, gamma = args.gamma).fit(show_plot = True)