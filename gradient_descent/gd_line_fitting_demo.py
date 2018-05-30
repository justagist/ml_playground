'''
    Gradient Descent Example
    
    @author: JustaGist
    @package: ml_playground

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation



class GradientDescent:
    '''
        Class for performing Gradient Descent on a set of 2d points by fitting a line on them

    '''

    def __init__(self, learning_rate, points, visualise = False, starting_b = 0.0, starting_m = 0.0, num_iterations = 1000):

        self.learning_rate_ = learning_rate
        self.points_ = points
        self.animate_ = visualise
        self.num_iterations_ = num_iterations

        self.b_ = starting_b # current y-intercept of the fitted line
        self.m_ = starting_m # current slope of the fitted line

        if self.animate_:
            self.initialise_plotter_()


    def initialise_plotter_(self):
        '''
            For matplotlib animation
        '''
        self.fig_ = plt.figure()
        self.ax_ = plt.axes(xlim=(0, 80), ylim=(20,120))
        self.line_, = self.ax_.plot([], [], lw=2) # ----- an Artist object that should be returned each time in the FuncAnimation function


    def compute_error(self):
        '''
            Computes the error between the points and the fitted line
        '''
        totalError = 0
    
        for i in range(0, len(self.points_)):
            x = self.points_[i, 0]
            y = self.points_[i, 1]
            totalError += (y - (self.m_ * x + self.b_)) ** 2
        
        return totalError / float(len(self.points_))

    def init_animation_(self):
        '''
            FuncAnimation requirement
        '''
        self.line_.set_data([], [])
        return self.line_,


    def step_gradient(self):
        '''
            Performs one gradient step. Checks the direction of the cost function by finding its derivative and steps towards the direction of lower cost according to the learning rate.
            self.b_ and self.m_ are updated accordingly.
        '''

        b_gradient = 0
        m_gradient = 0

        nppoints = np.array(self.points_)
        N = float(len(nppoints))
        
        for i in range(0, len(nppoints)):
            x = nppoints[i, 0]
            y = nppoints[i, 1]
            b_gradient += -(2/N) * (y - ((self.m_ * x) + self.b_))
            m_gradient += -(2/N) * x * (y - ((self.m_ * x) + self.b_))
        self.b_ = self.b_ - (self.learning_rate_ * b_gradient)
        self.m_ = self.m_ - (self.learning_rate_ * m_gradient)

    def get_line_points_(self):
        '''
            Fitting a line for the points according to the current m and b. Returns the Artist object required by the FuncAnimation function
        '''

        x = np.linspace(0, 100, 2)
        y = self.m_ * x + self.b_

        self.line_.set_data(x, y)
        return self.line_


    def step_gradient_and_get_line(self, i):

        self.step_gradient()
        line = self.get_line_points_()
        return line,

    def run(self):
        '''
            Perform gradient descent for the required number of iterations.
        '''
        if self.animate_:
            anim = animation.FuncAnimation(self.fig_, self.step_gradient_and_get_line, init_func=self.init_animation_,
                               frames=self.num_iterations_, interval=200, blit=True, repeat=False)

            plt.scatter(self.points_[:,0], self.points_[:,1], color='g') # Plotting the points
            plt.show()

        else:

            for i in range(self.num_iterations_):

                self.step_gradient()



if __name__ == '__main__':

    points = np.genfromtxt("data.csv", delimiter=",")

    gd = GradientDescent(learning_rate = 0.00001, points=points, visualise = True, starting_b = 0, starting_m = 0, num_iterations = 1000)

    print "Initial Error:", gd.compute_error()

    gd.run()

    print "Final Error:", gd.compute_error()
