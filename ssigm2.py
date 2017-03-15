import numpy as np
import pylab
import matplotlib.pyplot as plt

# Define sigmoid function
def sigm(x):
    # To prevent overflow, we use the following stable implementation
    return 1./(1. + np.exp(-x))

# Define function for generating target data from feature data
def gen_target(data, n_rows, a, b, r):
    r2 = r*r
    target = np.zeros(n_rows)
    for i in range(n_rows):
        temp = (data[i, 0] - a)**2 + (data[i, 1] - b)**2
        if temp < r2:
            target[i] += 1

    return target

# Define class to contain parameters for neural network
class Parm:
    # Initialize parameters
    def __init__(self, eta, b_size, data_rows, data_cols):
        # Initialize learning rate
        self.eta = eta

        # Initialize batch size
        self.batch_size = b_size

        # Initialize data row and column sizes
        self.rows = data_rows
        self.cols = data_cols

        # Initialize batch
        self.batch = np.random.randint(0, self.rows, self.batch_size)

    # Modify values in batch
    def shuffle(self):
        self.batch = np.random.randint(0, self.rows, self.batch_size)

# Define class to contain weights for neural network
class Weights:
    # Initialize parameters
    def __init__(self, data_cols, n_hidden, uniform):
        # Initialize number of hidden nodes
        self.n_hidden = n_hidden
        # Initialize either uniformly distributed weights or zeros
        if (uniform):
            # Initialize input bias terms 
            self.bi = np.random.uniform(-1, 1, n_hidden)

            # Initialize hidden bias term 
            self.bh = np.random.uniform(-1, 1)

            # Initialize input weights
            self.Wi = np.random.uniform(-.1, .1, (n_hidden, data_cols))

            # Initialize hidden weights
            self.Wh = np.random.uniform(-.1, .1, n_hidden)
        else:
            # Initialize input bias terms to zero
            self.bi = np.zeros(n_hidden)

            # Initialize hidden bias term 
            self.bh = 0

            # Initialize input weights to zeros
            self.Wi = np.zeros((n_hidden, data_cols))

            # Initialize hidden weights
            self.Wh = np.zeros(n_hidden)


class Network:
    # Initialize parameters
    def __init__(self, eta, b_size, data_rows, data_cols, n_hidden, uniform):
        self.w = Weights(data_cols, n_hidden, uniform)
        self.p = Parm(eta, b_size, data_rows, data_cols)

    # Define function to generate output of neural network
    def h_out(self, feature):
        # Compute value at which to evaluate sigmoid function
        x = np.dot(self.w.Wi, feature) + self.w.bi
        return sigm(x)

    def o_out(self, x):
        # Compute value at which to evaluate sigmoid function
        temp = np.dot(self.w.Wh, x) + self.w.bh
        return sigm(temp)

    # Define error function for neural network
    def err(self, target, output):
        return target - output

    # Define total error function for neural network
    def tot_err(self, feature, target, test_size):
        temp = 0

        for i in range(test_size):
            o_h = self.h_out(feature[i])
            o = self.o_out(o_h)
            temp += 0.5 * (target[i] - o)**2

        print(" Total error = ", temp)

        #return temp

    def update_Wh (self, Err, o, o_h):
        temp = self.p.eta * Err * o * (1 - o)
#        print("Wh temp = ", temp)
        self.w.bh += temp

        update = o_h * temp #np.multiply(o_h, temp)
#        print("Wh update = ", update)
        self.w.Wh += update #np.add(self.w.Wh, update)
#        for j in range(self.cols): #np.nditer(self.batch):
#            self.w.Wh[j] += temp * feature[j]

    def update_Wi (self, feature, Err, o, o_h):
#        print("Err = ", Err)
#        print("o = ", o)
#        print("1 - o = ", 1 - o)
#        print("o_h = ", o_h)
#        print("1 - o_h = ", 1 - o_h)
#        print("Err * o * (1 - o) * o_h[0] * (1 - o_h)[0] * Wh[0] = ", 
#               Err * o * (1 - o) * o_h[0] * (1 - o_h)[0] * self.w.Wh[0])
        temp = self.p.eta * Err * o * (1 - o) * self.w.Wh * o_h * (1 - o_h)
#        print("Wi temp = ", temp)
        self.w.bi = np.add(self.w.bi, temp)

        for j in range(self.w.n_hidden):
            update = feature * temp[j]
#            print("Wi update %d = " %j)
#            print(update)
            self.w.Wi[j] = np.add(self.w.Wi[j], update)
#        for j in range(self.cols): #np.nditer(self.batch):
#            self.w.Wh[j] += temp * feature[j]


    # Define function for training neural network on data set
    def train(self, feature, target):
        num_err = 0

        for i in range(self.p.rows):
            o_h = self.h_out(feature[i])
#            print("o_h = ", o_h)
            o = self.o_out(o_h)
            Err = self.err(target[i], o)
#            print(o)
#            print(Err)        

#            if (Err != 0):
#                num_err += 1
            self.update_Wh(Err, o, o_h)
            self.update_Wi(feature[i], Err, o, o_h)
        
        #print("training errors = ", num_err)

    # Define function for testing neural network on data set
    def test(self, feature, target, test_size, data_title):
        num_err = test_size

        for i in range(test_size):
            o_h = self.h_out(feature[i])
            o = self.o_out(o_h)
            Err = self.err(target[i], o)
#            print("o = ", o, ", t = ", target[i], ", and Err = ", Err)
            #if (np.abs(Err) < 0.5):
            #    num_err -= 1
            
            if(o >= 0.5):
                o = 1
            else:
                o = 0
            
            Err = self.err(target[i], o)
#            print("o = ", o, ", t = ", target[i], ", and Err = ", Err)

            if(target[i] == o):
                num_err -= 1  

        print(" Number of errors from", data_title, ":", num_err)

    # Define function for plotting test results from Nerual Network on test set
    def plot_test(self, feature, target, test_size, plot_title):
        x_no = []
        x_yes = []
        y_no = []
        y_yes = []

        for i in range(test_size):
            o_h = self.h_out(feature[i])
            o = self.o_out(o_h)
            
            # Store predictions based on model developed by Neural Network
            if (o >= 0.5):
                x_yes.append(feature[i, 0])
                y_yes.append(feature[i, 1])
            else:
                x_no.append(feature[i, 0])
                y_no.append(feature[i, 1])

        ax = plt.gca()
        ax.cla()        # Clear the grid for fresh plot

        plt.plot(x_no[:], y_no[:], 'ro', markersize=6, lw=3)
        plt.plot(x_yes[:], y_yes[:], 'go', markersize=6, lw=3)
        circle1 = plt.Circle((0.5, 0.6), 0.4, color='k', lw=3, fill=False)
        plt.gca().set_aspect('equal', adjustable='box')
        ax.add_artist(circle1)
        # set and label axes, set title, and legend
        plt.axis([0, 1, 0, 1])
        plt.title(plot_title)
        plt.xlabel('x')
        plt.ylabel('y')
#        plt.legend(bbox_to_anchor=(.826, .05), loc=2, borderaxespad=-1.)
        plt.grid(True)
        #plt.show()

        plot_title.__add__('.png')
        plt.savefig(plot_title)


# Initialize Neural Network
N = Network(1, 5, 100, 2, 10, True)

# Generate training data
train_feature = np.random.uniform(0, 1, (N.p.rows, N.p.cols))
train_target = gen_target(train_feature, N.p.rows, 0.5, 0.6, 0.4)

#print(train_feature)
#print(train_target)

# Generate testing data
testing_size = 100
test_feature = np.random.uniform(0, 1, (testing_size, N.p.cols))
test_target = gen_target(test_feature, testing_size, 0.5, 0.6, 0.4)


N.test(train_feature, train_target, N.p.rows, 'training data')
N.tot_err(train_feature, train_target, N.p.rows)

N.test(test_feature, test_target, testing_size, 'testing data')
N.tot_err(test_feature, test_target, testing_size)
#print(" Wh = \n", N.w.Wh)

for i in range(1000):
    N.train(train_feature, train_target)
    #N.tot_err(train_feature, train_target, N.p.rows)
    #N.tot_err(test_feature, test_target, testing_size)

N.test(train_feature, train_target, N.p.rows, 'training data')
N.tot_err(train_feature, train_target, N.p.rows)

N.test(test_feature, test_target, testing_size, 'testing data')
N.tot_err(test_feature, test_target, testing_size)

N.plot_test(train_feature, train_target, N.p.rows, 
            'Plot of Classification of Training Data')
N.plot_test(test_feature, test_target, testing_size, 
            'Plot of Classification of Testing Data')

