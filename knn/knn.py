import csv
import math
import random
import operator
 

class knnClassifier:

    ''' 
        Perform knn classification on the given file, which has num_attributes attributes per data point. 
        The dataset is split into training set and testing set with the raitio split_ratio. 
        The classifier will classify the test set into k classes.

    '''

    def __init__(self, filename, num_attributes = 4, split_ratio = 0.65):

        self.num_attributes_ = num_attributes # ----- number of attributes for each data point

        self.training_set_ = []
        self.testing_set_ = []

        self.load_dataset_from_csv(filename, split_ratio)


    def load_dataset_from_csv(self, filename, split_ratio):

        with open(filename, 'rb') as datafile:

            lines = csv.reader(datafile)
            dataset = list(lines)

            for x in range(len(dataset)-1):

                for y in range(self.num_attributes_):
                    dataset[x][y] = float(dataset[x][y]) # ----- convert all values to floats

                # ----- split the dataset to training and testing set depending on the split ratio
                if random.random() < split_ratio: 
                    self.training_set_.append(dataset[x])
                else:
                    self.testing_set_.append(dataset[x])

        print '\nTraining Set: %d'%len(self.training_set_)
        print 'Testing Set: %d\n'%len(self.testing_set_)

    def classify(self, k, verbose = True):
        '''
            Classify the test set into k classes using k-nearest neighbour method.

        '''
        predictions = []

        for x in range(len(self.testing_set_)):

            neighbors = self._get_k_nearest_neighbours(self.testing_set_[x], k)

            prediction = self._find_majority_class(neighbors)

            predictions.append(prediction)

            if verbose:
                print('%d. Prediction = %s \t;\tactual = %s'%(x+1, prediction, self.testing_set_[x][-1]))

        return predictions



    def _euclidean_distance(self, instance_1, instance_2):
        '''
            Find Euclidean distance between two instances as a measure of the sum of the squared differences between their corresponding attributes.

        '''
        distance = 0

        for x in range(self.num_attributes_):
            distance += pow((instance_1[x] - instance_2[x]), 2)

        return math.sqrt(distance)
 
    def _get_k_nearest_neighbours(self, test_instance, k):
        '''
            Get k data points from the training set that are the closest to given test instance.

        '''

        distances = []

        for x in range(len(self.training_set_)):
            # ----- find ethe euclidean distance of each data point and add the values to distances
            dist = self._euclidean_distance(test_instance, self.training_set_[x])
            distances.append((self.training_set_[x], dist))

        # ----- sort the data points in decreasing order of distances
        distances.sort(key=operator.itemgetter(1))

        neighbors = []

        # ----- add the first k datapoints to neighbours
        for x in range(k):
            neighbors.append(distances[x][0])

        return neighbors

    def _find_majority_class(self, neighbors):
        '''
            Finds the most popular class in the given set of neighbours.
            
        '''

        class_votes = {} # ----- create a dictionary to hold the class names and the number of votes they get.

        for x in range(len(neighbors)):

            response = neighbors[x][-1] # ----- The class to which this neighbour belongs to.

            # ----- If that class is already in the dictionary, add a vote to it.
            if response in class_votes:
                class_votes[response] += 1

            # ----- If the class is not in the dictionary, add the key and give it the first vote.
            else:
                class_votes[response] = 1

        # ----- sort the dictionary in the decreasing order of votes.
        sorted_votes = sorted(class_votes.iteritems(), key=operator.itemgetter(1), reverse=True)

        # ----- return the name of the class with the highest votes.
        return sorted_votes[0][0]

    def check_accuracy(self, predictions):
        '''
            Compares the predictions and actual classes of the test set to determine classification accuracy.
        '''
        correct_predictions = 0

        for x in range(len(self.testing_set_)):

            if self.testing_set_[x][-1] == predictions[x]:
                correct_predictions += 1

        return (correct_predictions/float(len(self.testing_set_))) * 100.0



if __name__ == '__main__':
    
    ''' 
        Perform knn classification on the iris.data file, which has 4 attributes per data point. 
        The dataset is split into training set and testing set with the raitio 0.65. 
        The classifier is supposed to classify the test set into 3 classes (k = 3)

    '''
    classifier = knnClassifier(filename = 'iris.data', num_attributes = 4, split_ratio = 0.65)

    # ----- The iris dataset consists of data of k = 3 different classes.
    predictions = classifier.classify(k = 3, verbose = True)

    print "\nAccuracy of prediction: %f%%\n"%classifier.check_accuracy(predictions)

