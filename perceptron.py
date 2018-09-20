import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, n=1000, learning_rate=0.05):
        '''
        Objective: To initialize data memebers
        Input Parameters: s
            self - object of type Perceptron
            no_of_inputs, epochs, learning_rate - int
        Return Value: None
        '''
        self.no_of_inputs = no_of_inputs
        self.epochs = n
        self.learning_rate = learning_rate
        self.weights =0.05 * np.random.randn(self.no_of_inputs + 1)# weights corr to num_inputs + 1 for bias 
        print("Initial values of weights",self.weights)
    def predict(self, inputs):
        '''
        Objective: To predict the output
        Input Parameters: 
            self - object of type Perceptron
            inputs - numpy array
        Return Value: Binary
        '''
        summation = np.dot(self.weights[1:].T, inputs) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        '''
        Objective: To train the model so as to tune weights
        Input Parameters:  
                self - object of type Perceptron
                training_inputs, labels - np array
        Return Value: None
        '''
        for _ in range(self.epochs):
            deltaWeights = np.array([0] * training_inputs.shape[1]).astype('float64')
            deltaBias = 0.0
            nInstances = training_inputs.shape[0]
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                deltaWeights += self.learning_rate * (label - prediction) * inputs
                deltaBias += self.learning_rate * (label - prediction)
            self.weights[1:] += (1/nInstances)*deltaWeights
            self.weights[0] += (1/nInstances)*deltaBias