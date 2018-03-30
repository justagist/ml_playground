'''
    Training and applying a Support Vector Machine using a simple network on the given linearly separable data.
    
    @author: JustaGist
    @package: ml_playground

    @Usage: python run_svm.py --train linearly_separable_data.csv --svmC 1 --verbose True --num_epochs 10
            python run_svm.py -h

'''

import sys
import tensorflow as tf
import numpy as np
import scipy.io as io
from matplotlib import pyplot as plt
from matplotlib import colors


def read_csv(filename):

    out = np.loadtxt(filename, delimiter=',');

    # Arrays to hold the labels and feature vectors.
    labels = out[:,0]
    labels = labels.reshape(labels.size,1)
    fvecs = out[:,1:]

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs,labels


class SupportVectorMachine:

    def __init__(self, train_data, svmC = 1, batch_size = 100):

        self.data_ = train_data

        self.train_size_, self.num_features_ = self.data_.shape

        self.svmC_ = svmC
        self.batch_size_ = batch_size

        self._initialize_model()

    def _initialize_model(self):

        # ----- This is where training samples and labels are fed to the graph.
        self.x_ = tf.placeholder("float", shape=[None, self.num_features_]) # ----- input points
        self.y_ = tf.placeholder("float", shape=[None,1]) # ----- true output class

        self._create_network()
        self._define_loss()
        self._define_evaluator()

    def _create_network(self):
        '''
            Define and initialize the network.

        '''
        # ----- These are the weights and bias that inform how much each feature contributes to the classification.
        self.w_ = tf.Variable(tf.zeros([self.num_features_,1]))
        self.b_ = tf.Variable(tf.zeros([1]))
        self.y_raw_ = tf.matmul(self.x_, self.w_) + self.b_ 

    def _define_loss(self):

        regularization_loss = 0.5*tf.reduce_sum(tf.square(self.w_)) 

        hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([self.batch_size_,1]), 1 - self.y_*self.y_raw_));

        self.svm_loss_ = regularization_loss + self.svmC_*hinge_loss;

    def _define_evaluator(self):

        self.predicted_class_ = tf.sign(self.y_raw_);

        correct_prediction = tf.equal(self.y_,self.predicted_class_)

        self.accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def train(self, labels, epochs = 5, lr = 0.01, verbose = False, plot = True):
        '''
            Train on the initialised data using the provided labels and training parameters

        '''

        def define_optimiser():
            return tf.train.GradientDescentOptimizer(lr).minimize(self.svm_loss_)

        train_step = define_optimiser()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        if verbose:
            print 'Initialized!'
            print
            print 'Training.'

        # Iterate and train.
        for step in xrange(epochs * self.train_size_ // self.batch_size_):
            if verbose:
                print step+1,
                
            offset = (step * self.batch_size_) % self.train_size_
            batch_data = self.data_[offset:(offset + self.batch_size_), :]
            batch_labels = labels[offset:(offset + self.batch_size_)]
            _, loss = sess.run([train_step, self.svm_loss_], feed_dict={self.x_: batch_data, self.y_: batch_labels})
            print 'loss: ', loss
            
            if verbose and offset >= self.train_size_-self.batch_size_:
                print

        # Give very detailed output.
        if verbose:
            print
            print 'Weight matrix.'
            print sess.run(self.w_)
            print
            print 'Bias vector.'
            print sess.run(self.b_)
            print
            print "Applying model to first test instance."
            print
            
        print "Accuracy on train: %f%%"%(sess.run(self.accuracy_, feed_dict={self.x_: self.data_, self.y_: labels})*100)

        if plot:
            eval_fun = lambda X: sess.run(self.predicted_class_,feed_dict={self.x_:X}); 
            self.plot_svm(self.data_, labels.flatten(), eval_fun)

    def plot_svm(self, X, Y, pred_func):
        # determine canvas borders
        mins = np.amin(X,0); 
        mins = mins - 0.1*np.abs(mins);
        maxs = np.amax(X,0); 
        maxs = maxs + 0.1*maxs;

        ## generate dense grid
        xs,ys = np.meshgrid(np.linspace(mins[0],maxs[0],300), np.linspace(mins[1], maxs[1], 300));


        # evaluate model on the dense grid
        Z = pred_func(np.c_[xs.flatten(), ys.flatten()]);
        Z = Z.reshape(xs.shape)

        # Plot the contour and training examples
        plt.contourf(xs, ys, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap=colors.ListedColormap(['r', 'b']))
        plt.show()



def run_SVM(argv=None):

    verbose = FLAGS.verbose
    plot = FLAGS.plot
    
    # Get the data.
    train_data_filename = FLAGS.train

    # Extract it into numpy matrices.
    train_data,train_labels = read_csv(train_data_filename)

    # Convert labels to +1,-1
    train_labels[train_labels==0] = -1

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # Get the C param of SVM
    svmC = FLAGS.svmC

    batch_size = FLAGS.batch_size
    lr = FLAGS.lr

    SupportVectorMachine(train_data, svmC, batch_size).train(train_labels, num_epochs, lr, verbose, plot)

    
if __name__ == '__main__':

    # ===================== Define the flags useable from the command line =============================

    tf.app.flags.DEFINE_string('train', None, 'File containing the data (2D points with labels).')
    tf.app.flags.DEFINE_integer('num_epochs', 1, 'Number of training epochs.')
    tf.app.flags.DEFINE_integer('batch_size', 100, 'Number of training points to use per training step')
    tf.app.flags.DEFINE_float('svmC', 1, 'The C parameter of the SVM cost function.')
    tf.app.flags.DEFINE_float('lr', 0.01, 'Learning rate for training.')
    tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
    tf.app.flags.DEFINE_boolean('plot', True, 'Plot the final decision boundary on the data.')

    FLAGS = tf.app.flags.FLAGS

    if FLAGS.train is None:
        sys.exit("\n       Data not provided! Run code with -h/--help flag to see usage.\n\nEXITING!\n")


    # ==================================================================================================

    tf.app.run(main=run_SVM)
