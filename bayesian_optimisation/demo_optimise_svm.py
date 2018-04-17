import numpy as np
import plotter
import bayesian_optimisation
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn import datasets, model_selection, svm

# data, target = datasets.make_classification(n_samples=500,
#                                    n_features=45,
#                                    n_informative=15,
#                                    n_redundant=25)

# def sample_loss(params):
#     return model_selection.cross_val_score(svm.SVC(C=10 ** params[0], gamma=10 ** params[1], random_state=12345),
#                            X=data, y=target, scoring='roc_auc', cv=3).mean()


# lambdas = np.linspace(1, -4, 25)
# gammas = np.linspace(1, -4, 20)

# # We need the cartesian combination of these two vectors
# param_grid = np.array([[C, gamma] for gamma in gammas for C in lambdas])

# real_loss = [sample_loss(params) for params in param_grid]

# # The maximum is at:
# param_grid[np.array(real_loss).argmax(), :]

# from matplotlib import rc
# rc('text', usetex=True)

# C, G = np.meshgrid(lambdas, gammas)
# # plt.figure()
# # cp = plt.contourf(C, G, np.array(real_loss).reshape(C.shape))
# # plt.colorbar(cp)
# # plt.title('Filled contours plot of loss function $\mathcal{L}$($\gamma$, $C$)')
# # plt.xlabel('$C$')
# # plt.ylabel('$\gamma')
# # # plt.savefig('/Users/thomashuijskens/Personal/gp-optimisation/figures/real_loss_contour.png', bbox_inches='tight')
# # plt.show()


# bounds = np.array([[-4, 1], [-4, 1]])

# xp, yp = bayesian_optimisation.bayesian_optimisation(n_iters=30, 
#                                sample_loss=sample_loss, 
#                                bounds=bounds,
#                                n_pre_samples=3,
#                                random_search=100000)


# rc('text', usetex=False)
# plotter.plot_iteration(lambdas, xp, yp, first_iter=3, second_param_grid=gammas, optimum=[0.58333333, -2.15789474])


import imageio
images = []

for i in range(3, 23):
    filename = "figs/bo_iteration_%d.png" % i 
    images.append(imageio.imread(filename))
    
imageio.mimsave('figs/bo_2d_new_data.gif', images, duration=1.0)