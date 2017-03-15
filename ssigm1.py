import numpy as np
import csv
#np.float128
#np.seterr(over='raise')

# Define sigmoid function
def sigm(x):
    # To prevent overflow, we use the following stable implementation
    if x >= 0:
        return 1./(1. + np.exp(-x))
    else:
        return np.exp(x)/(1. + np.exp(x))

# Define class to contain parameters for neural network
class Parm:
    # Initialize parameters
    def __init__(self, b_size, data_rows, data_cols, uniform):
        # Initialize learning rate
        self.eta = 1

        # Initialize data row and column sizes
        self.rows = data_rows
        self.cols = data_cols

        # Initialize batch size
        self.batch_size = b_size

        # Initialize batch
        self.batch = np.random.randint(0, self.rows, self.batch_size)

        # Initialize either uniformly distributed weights or zeros
        if (uniform):
            d = 4 * np.sqrt(6/(1 + self.cols))
            # Initialize bias term 
            self.b = np.random.uniform(-d, d)

            # Initialize weights
            self.arr = np.random.uniform(-d, d, data_cols)
        else:
            # Initialize bias term to zero
            self.b = 0

            # Initialize weights to zeros
            self.arr = np.zeros(data_cols)

    # Modify values in batch
    def shuffle(self):
        self.batch = np.random.randint(0, self.rows, self.batch_size)

    # Define function to generate output of neural network
    def out(self, feature):
        # Compute value at which to evaluate sigmoid function
        temp = np.dot(feature, self.arr) + self.b
        return np.tanh(temp) #sigm(temp)

    # Define error function for neural network
    def err(self, target, output):
        return target - output

    def tot_err(self, feature, target, indices):
        temp = 0

        for i in np.nditer(indices):
            out = self.out(feature[i])
            temp += 0.5 * (target[i] - out)**2

        return temp

    # Define function to update weights based on error
    def batch_update_w (self, feature, err, out):
        temp = self.eta * err * out * (1 - out) / self.batch_size

        for i in range(self.cols): #np.nditer(self.batch):
            if (i != -1):
                self.arr[i] += temp * feature[i]
            else:
                self.b += temp

    # Define function for training neural network on data set
    def nnet_train(self, feature, target):
        num_err = 0

        for i in np.nditer(self.batch):
            out = self.out(feature[i])
            err = self.err(target[i], out)
        
            if (err != 0):
                num_err += 1
                self.batch_update_w(feature[i], err, out)
        
        #print("training errors = ", num_err)

    # Define function for testing neural network on data set
    def nnet_test(self, feature, target):
        num_err = 0

        for i in range(self.rows):
            out = self.out(feature[i])

            if (target[i] != out):
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
train = np.genfromtxt('bank_numeric.csv', skip_header=1, 
                       skip_footer=2260, delimiter=';')

#print(train[0])
#print(train_feature[0])
#print(train_target[0])
#print(train_data[2233])

# The array test_data has 2233 rows and 44 columns.
test_feature = np.genfromtxt('bank_feature.csv', skip_header=2262, delimiter=';')
test_target = np.genfromtxt('bank_target.csv', skip_header=2262, delimiter=';')

# Initialize weights for neural network
w = Parm(10, 2233, 44, True)
a = range(0, w.rows - 1)
#print(w.test_ret(test_feature[0]))

# Print initial parameters for neural network
print("\n------------------------ Initial Weights ------------------------")
print(" Initial bias = ", w.b)
print(" Initial w = \n", w.arr)

print("\n----------------------- Initial Parameters -----------------------")
print(" Learning rate = ", w.eta)
print(" Batch size = ", w.batch_size)

print("\n------------------------- Initial Errors -------------------------")
print(" Error function for training values at start: ",
      w.tot_err(train_feature,train_target,a))
init_err = w.nnet_test(train_feature, train_target)
print(" Percent of misclassified training features: "
      "%3.4lg%%" % (100 * init_err/2262))
print(" Error function for testing values at start: ",
      w.tot_err(test_feature,test_target,a))
init_err = w.nnet_test(test_feature, test_target)
print(" Percent of misclassified testing features: "
      "%3.4lg%%" % (100 * init_err/2262))

# Train over 10 epochs
for i in range(50):
    w.shuffle()
    w.nnet_train(train_feature, train_target)
    temp = w.tot_err(train_feature,train_target,a)
#    print("Error at step %d: %lg\n" %(i, temp))

print("\n------------------------- Final Weights -------------------------")
print(" Final bias = ", w.b)
print(" Final w = \n", w.arr)

print("\n------------------------- Final Errors -------------------------")
print(" Error function for training values at finish: ",
      w.tot_err(train_feature,train_target,a))
fin_err = w.nnet_test(train_feature, train_target)
print(" Percent of misclassified training features: "
      "%3.4lg%%\n" % (100 * fin_err/2262))
print(" Error function for testing values at finish: ",
      w.tot_err(test_feature,test_target,a))
fin_err = w.nnet_test(test_feature, test_target)
print(" Percent of misclassified testing features: "
      "%3.4lg%%" % (100 * fin_err/2262))
