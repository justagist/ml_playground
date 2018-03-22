'''
    Gradient Descent Example

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

global_points = np.genfromtxt("data.csv", delimiter=",")
learning_rate = 0.0001

fig = plt.figure()
ax = plt.axes(xlim=(0, 80), ylim=(20,120))
line, = ax.plot([], [], lw=2)


def compute_error(b, m, points):
    
    totalError = 0
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    
    b_gradient = 0
    m_gradient = 0
    nppoints = np.array(points)
    N = float(len(nppoints))
    
    for i in range(0, len(nppoints)):
        x = nppoints[i, 0]
        y = nppoints[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)

    return [new_b, new_m]

def close_event():
    plt.close()

def animate(i):
    global global_b
    global global_m
    
    global_b, global_m = step_gradient(global_b, global_m, global_points, learning_rate)

    x = np.linspace(0, 100, 100)
    y = global_m * x + global_b

    line.set_data(x, y)
    return line,


def perform_gradient_descent(starting_b, starting_m, num_iterations):
    global global_b
    global global_m
    global_b = starting_b
    global_m = starting_m

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=num_iterations, interval=200, blit=True)
    
    plt.scatter(global_points[:,0], global_points[:,1])
    plt.show()

def run():
    
    
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, global_points))
    
    print "Running..."
    
    perform_gradient_descent(initial_b, initial_m, num_iterations)
    
    # print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b, m, global_points))
    # plt.show()

def init():
    line.set_data([], [])
    return line,


if __name__ == '__main__':
    run()
