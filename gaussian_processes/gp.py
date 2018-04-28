'''
    Testing Gaussian Process Regression on simple artificial data
    
    @author: JustaGist
    @package: ml_playground

    @Usage: python gp.py

'''
import sys
import numpy as np
import matplotlib.pyplot as plt

class GaussianProcess:

    """
        Simple GP Regression assuming a zero mean prior.

    """


    def __init__(self, X, Y, data_noise = 0.00005, **kwargs):

        '''
            Each datapoint is a function with the mean at f(x) and an uncertainty (if the data is noisy).

            Args:
                X: n training points of dim D
                Y: n values corresponding to f(X), where f is the true function that is to be approximated
                data_noise: the variance of the gaussian noise assumed to be present in the data
                kwargs:
                    kernel_parameter: the parameter that determines the 'minimum distance' required between datapoints to be considered as nearby points

        '''

        self.X_ = X # ----- contains the input data in shape N x D, where N is the number of training points, and D their dimensions.
        self.Y_ = Y # ----- the value corresponding to each training point. 

        self.data_noise_ = data_noise # ----- the variance of the assumed gaussian noise in the data 

        self.init_model(**kwargs) # ----- create the prior distribution using the given training points



    def init_model(self, **kwargs):
        '''
            Creates the initial GP model for the given training points.

            kwargs:
                kernel_parameter: the parameter that determines the 'minimum distance' required between datapoints to be considered as nearby points
        '''

        self.mu_ = self.Y_ # ----- Initially each Y will be the mean of the function at that point (variance will be computed).

        self.sigma_ = self.compute_covariance(self.X_, self.X_, **kwargs) # ----- The covariance matrix
        self.L_ = self.cholesky_decomposition(self.sigma_, noise = self.data_noise_) # ----- The cholesky decomposition of the covariance matrix, can be considered equivalent to the sqrt of the covariance matrix (std deviation). 


    def compute_covariance(self, a, b, **kwargs):

        '''
            Computes the covariance of the two matrices a and b
            Different metrics can be used to compute the covariance. Here the measure of how close the datapoints are in the domain (i.e. the x value) determines the measure of closeness. Intuitively, points which are nearby in the input space will have y values which are close by.

            Args:
                a,b: datapoints whose covariance is to be computed
                kwargs:
                    kernel_parameter: the parameter that determines the 'minimum distance' required between datapoints to be considered as nearby points

        '''
        # ----- Define the kernel
        def kernel(kernel_parameter = 0.5):
            ''' 
                GP squared exponential kernel. This kernel measures the closeness of the points in the input space. If they are close in the input space, their output values will be close as well.

            '''

            kp = kernel_parameter
            sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
            return np.exp(-.5 * (1/kp) * sqdist)

        cov = kernel(**kwargs)

        return cov

    def cholesky_decomposition(self, matrix, noise):

        '''
            Calculate the left triangular matrix for the given matrix using 'noisy' cholesky decomposition.
        '''
        assert matrix.shape[0] == matrix.shape[1]

        L = np.linalg.cholesky(matrix + noise*np.eye(matrix.shape[0]))

        return L


    def predict_model(self, testpoints, plot = False, **kwargs):
        '''
            Compute the mean and variance of the given test points using the training points

                kwargs can contain the keyword-argument 'true_values' containing the actual y values of the testpoints, which when given to plot_model function would plot the true function to be predicted at each test point.

        '''

        # ----- compute the mean at the test points. (according to de Freitas' slides)
        Lk = np.linalg.solve(self.L_, self.compute_covariance(self.X_, testpoints))
        mu = np.dot(Lk.T, np.linalg.solve(self.L_, self.Y_))

        # ----- compute the variance at our test points.
        K = self.compute_covariance(testpoints, testpoints)

        if plot:
            s2 = np.diag(K) - np.sum(Lk**2, axis=0)
            s = np.sqrt(s2)
            self.plot_model(test_X = testpoints, test_X_mu = mu, test_X_sigma = s, title = "Model Prediction", **kwargs)

        return mu, K, Lk # ----- return Lk for computing posterior (according to equation in de Freitas' slides)



    def plot_model(self, test_X, test_X_mu, test_X_sigma, true_values = None, title = None):

        plt.figure()
        plt.clf()

        # ----- plot the training points 
        plt.plot(self.X_, self.Y_, 'r+', ms=20)

        if true_values is not None:
            # ----- plot the test points and their actual value (which are the values to be predicted)
            plt.plot(test_X, true_values, 'b-')

        # ----- plot the mean and variance of the predictions made at the testpoints
        plt.gca().fill_between(test_X.flat, test_X_mu-3*test_X_sigma, test_X_mu+3*test_X_sigma, color="#dddddd")
        plt.plot(test_X, test_X_mu, 'r--', lw=2)

        if title is not None:
            plt.title(title)

        # plt.axis([-5, 5, -3, 3])
        plt.axis('equal')

        # plt.show()

    def sample_from_prior(self, testpoints, num_samples = 10, plot = True, alpha = 1e-6):

        # ----- draw samples from the prior at the test points.
        _, K, _ = self.predict_model(testpoints)

        L = np.linalg.cholesky(K + alpha*np.eye(testpoints.shape[0])) # ----- K = LL'
        f_prior = np.dot(L, np.random.normal(size=(testpoints.shape[0], num_samples))) # ----- f_prior = L*N(0,I)

        if plot:
            plt.figure()
            plt.clf()
            plt.plot(Xtest, f_prior)
            plt.title('Samples from the GP prior')
            plt.axis('equal')

            # plt.show()

    def sample_from_posterior(self, testpoints, num_samples = 10, plot = True, alpha = 1e-6):
        '''
            Sampling from the posterior requires the mean and covariance of the test points. Using Lk is for easier computation of the new means and covariance.

        '''
        # ---- draw samples from the posterior at the test points.
        mu, K, Lk = self.predict_model(testpoints)

        L = np.linalg.cholesky(K + alpha*np.eye(testpoints.shape[0]) - np.dot(Lk.T, Lk))
        f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(testpoints.shape[0], num_samples))) # ----- mu + L*N(0,I)

        if plot:
            plt.figure()
            plt.clf()
            plt.plot(testpoints, f_post)
            plt.title('Samples from the GP posterior')
            plt.axis('equal')

            # plt.show()



        

if __name__ == '__main__':

    # =================================================================================

    # ----- The true unknown function we are trying to approximate --------------
    f = lambda x: np.sin(0.9*x).flatten()

    if len(sys.argv) > 1:
        if sys.argv[1] == '2':
            f = lambda x: (0.25*(x**2)).flatten()

    # ----------------------------------------------------------------------------
    

    N = 10         # ----- number of training points.
    n = 50         # ----- number of test points.
    s = 0.00005    # ----- noise variance.

    # ----- sample some input points and noisy versions of the function evaluated at these points, i.e. training points
    X = np.random.uniform(-5, 5, size=(N,1))
    y = f(X) + s*np.random.randn(N)

    # ----- points to make predictions at, i.e. test points
    Xtest = np.linspace(-5, 5, n).reshape(-1,1)

    # =================================================================================


    # ----- create the GP model using the training points
    gp = GaussianProcess(X, y, kernel_parameter = 0.5)

    # ----- predict the function values at the test points using the gaussian models
    gp.predict_model(Xtest, plot = True, true_values = f(Xtest))

    # ----- sample values for the test points before and after updating the GP with the 
    gp.sample_from_prior(Xtest)
    gp.sample_from_posterior(Xtest)

    plt.show()

