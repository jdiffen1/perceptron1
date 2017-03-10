import numpy as np
#from numpy import genfromtxt
import csv

class Parm:
    def __init__(self):
        # Initialize learning rate
        self.eta = 1

class Weight:
    # Initialize either uniformly distributed weights or zeros
    def __init__(self, n, uniform):
        if (uniform):
            # Initialize bias term 
            self.b = np.random.uniform(-1, 1)

            # Initialize weights
            self.arr = np.random.uniform(-.1, .1, n)
        else:
            # Initialize bias term to zero
            self.b = 0

            # Initialize weights to zeros
            self.arr = np.zeros(n)

    # Define function to generate output of perceptron
    def percep_out(self, feature): #, weight, bias):
        return np.sign(np.dot(feature, self.arr) + self.b)

    # Define error function for perceptron
    def percep_err(self, target, output):
        return target - output
    #return np.sign(temp) #+ int(temp == 0)

    # Define function to update weights based on error
    def update_w (self, feature, err, eta):
        self.b += eta * err
        self.arr = np.add(self.arr, np.dot(eta * err, feature))

    # Define function for training perceptron on data set
    def percep_train(self, feature, target, eta, n):
        num_err = 0

        for i in range(n):
            out = self.percep_out(feature[i])
            err = self.percep_err(target[i], out)
        
            if (err != 0):
                num_err += 1
                self.update_w(feature[i], err, eta)
        
        #print("training errors = ", num_err)

    # Define function for testing perceptron on data set
    def percep_test(self, feature, target, n):
        num_err = 0

        for i in range(n):
            out = self.percep_out(feature[i])
            err = self.percep_err(target[i], out)
            if (err != 0):
                num_err += 1
  
        print(" Number of testing errors = ", num_err)

        return num_err


# Convert data file 'bank_numeric.csv' into two arrays: the first for
# training and the second for testing.

# The array train_feature has 2234 rows and 44 columns. 
train_feature = np.genfromtxt('bank_feature.csv', skip_header=1, 
                               skip_footer=2260, delimiter=';')
train_target = np.genfromtxt('bank_target.csv', skip_header=1, 
                               skip_footer=2260, delimiter=';')
#print(train_feature[0])
#print(train_target[0])
#print(train_data[2233])

# The array test_data has 2233 rows and 44 columns.
test_feature = np.genfromtxt('bank_feature.csv', skip_header=2262, delimiter=';')
test_target = np.genfromtxt('bank_target.csv', skip_header=2262, delimiter=';')

# Initialize weights for perceptron
w = Weight(44, True)

# Print initial weights for perceptron
print("\n------------------------- Initial Weights -------------------------")
print(" bias = ", w.b, "\n")
print(" w = ", w.arr)

# Initialize parameters for perceptron
p = Parm()
print("\n------------------------- Parameters -------------------------")
print(" Learning rate = ", p.eta, "\n")


print("\n------------------------- Initial Errors -------------------------")
init_err = w.percep_test(test_feature, test_target, 2232)
print(" Percent of misclassified features: %3.4lg%%" % (100 * init_err/2262))

# Train over 100 epochs
for i in range(100):
    w.percep_train(train_feature, train_target, p.eta, 2233)

print("\n------------------------- Final Weights -------------------------")
print(" bias = ", w.b, "\n")
print(" w = ", w.arr)

print("\n------------------------- Final Errors -------------------------")
fin_err = w.percep_test(test_feature, test_target, 2232)
print(" Percent of misclassified features: %3.4lg%%" % (100 * fin_err/2262))

