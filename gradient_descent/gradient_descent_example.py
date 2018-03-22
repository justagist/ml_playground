'''
    Gradient Descent Example

'''

import numpy as np
import matplotlib.pyplot as plt
# y = mx + b
# m is slope, b is y-intercept
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


    ## ----- uncomment to visualise (visualisation not good at all)
    # fig = plt.figure()
    # timer = fig.canvas.new_timer(interval = 500) #creating a timer object and setting an interval of 3000 milliseconds
    # timer.add_callback(close_event)
    # x1 = np.linspace(0, 70, 100)
    # y1 = new_m * x1 + new_b
    # plt.plot(x1,y1)
    # plt.scatter(points[:,0], points[:,1])
    # plt.ylabel('some numbers')

    # timer.start()
    # plt.show()

    ## --------------------

    return [new_b, new_m]

def close_event():
    plt.close()

def perform_gradient_descent(points, starting_b, starting_m, learning_rate, num_iterations):
    
    b = starting_b
    m = starting_m
    
    for i in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)

    return [b, m]

def run():
    
    points = np.genfromtxt("data.csv", delimiter=",")
    
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, points))
    
    print "Running..."
    
    [b, m] = perform_gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)
    
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b, m, points))
    # plt.show()


if __name__ == '__main__':
    run()
