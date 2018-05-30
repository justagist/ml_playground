'''
    Gradient Descent Example - cost visualisation
    
    @author: JustaGist
    @package: ml_playground

'''

import numpy as np
import matplotlib.pyplot as plt 
import time


class GradDescent2D:

    def __init__(self, loss_function, function_gradient, xlims = [-20,20], ylims = [-20,20], axis_steps = 50j):


        self._loss_function = loss_function
        self._gradient_function = function_gradient

        self._initialise_contour(xlims, ylims, axis_steps)



    def _initialise_contour(self, xlims, ylims, steps):

        y, x = np.mgrid[xlims[0]:xlims[1]:steps, ylims[0]:ylims[1]:steps]

        f_vals = self._loss_function(x,y)
        grad_x, grad_y = self._gradient_function(x,y)

        self._create_canvas(x,y, f_vals, grad_x, grad_y)


    def _create_canvas(self, x, y, f_values, grad_x, grad_y, cmap = "Reds"):

        fig = plt.figure(figsize = (10,10))
        self._ax = fig.add_subplot(111)
        plt.ion()

        cp = self._ax.contourf(x, y, f_values, cmap = cmap)
        plt.colorbar(cp)

        plt.quiver(x[::3, ::3], y[::3, ::3],
               -grad_x[::3, ::3], -grad_y[::3, ::3],
               scale_units = "inches", pivot = "mid")
        plt.axis("image")

        plt.xlabel('x')
        plt.ylabel('y')

        plt.draw()


    def _draw_on_canvas(self, hist):

        self._ax.plot(hist[:,0], hist[:,1], 'bo-', linewidth = 4, ms = 13)

        self._ax.plot(hist[hist.shape[0]-1,0], hist[hist.shape[0]-1,1], 'r*', ms = 13)

        plt.draw()
        plt.pause(1)


    def descent(self, start_x, start_y, lr = 0.05, num_iter = 100, min_cost = 0.001):

        curr_x, curr_y = start_x, start_y

        w_hist = np.array([[curr_x, curr_y]])

        i = 0

        while i < num_iter and np.linalg.norm(self._loss_function(curr_x, curr_y)) > min_cost: 

            grad_fx, grad_fy = self._gradient_function(curr_x,curr_y)

            curr_x -= lr * grad_fx
            curr_y -= lr * grad_fy

            w_hist = np.append(w_hist, np.array([[curr_x, curr_y]]), axis=0)

            self._draw_on_canvas(w_hist)

            i+=1
            print i, "x:", curr_x, "y:", curr_y, "\tcost:", self._loss_function(curr_x, curr_y)



#######################
###### DEMO CODE ######
#######################

def f(x1,x2):
    return 2*x1**2 + x2**2 + 3*x1*x2 + 4

def grad_f(x1,x2):
    return 4*x1 + 3*x2, 2*x2 + 3*x1

if __name__ == '__main__':

    gd2d = GradDescent2D(loss_function = f, function_gradient = grad_f)

    gd2d.descent(start_x = 15, start_y = 19, lr = 0.05, num_iter = 100, min_cost = 0.001)
    
