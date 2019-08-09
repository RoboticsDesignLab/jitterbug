import tensorflow as tf
import numpy as np
import scipy.io as sio


class Autoencoder():
    """An autoencoder to reduce the input dimension and control the jitterbug in a smaller state space."""

    def __init__(self,
                 data,
                 splitting_percentage=0.8,
                 lr=0.01,
                 ):
        self.data = data
        self.N = len(data)
        self.splitting_percentage = splitting_percentage

        # Split the data into training and testing sets
        self.splitting_int = int(round(self.splitting_percentage*self.N,0))
        self.training_data = data[:self.splitting_int]
        self.testing_data = data[self.splitting_int:]

        self.lr = lr

        # Build the autoencoder
        num_inputs = len(data[0])
        num_hid1 = 4
        num_outputs = num_inputs
        actf = tf.nn.relu
        self.X = tf.placeholder(tf.float32, shape=[None, num_inputs])
        initializer = tf.variance_scaling_initializer()

        w1 = tf.Variable(initializer([num_inputs, num_hid1]), dtype=tf.float32)
        w2 = tf.Variable(initializer([num_hid1, num_outputs]), dtype=tf.float32)

        b1 = tf.Variable(tf.zeros(num_hid1))
        b2 = tf.Variable(tf.zeros(num_outputs))

        self.hid_layer1 = actf(tf.matmul(self.X, w1) + b1)
        self.output_layer = actf(tf.matmul(self.hid_layer1, w2) + b2)

        self.loss = tf.reduce_mean(tf.square(self.output_layer - self.X))

        optimizer = tf.train.AdamOptimizer(lr)
        self.train = optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()

        self.predictions = None

    def train_autoencoder(self,
              num_epoch=5,
              batch_size=150
              ):
        """Train the autoencoder usinf self.training_data.

        Args:
            num_epoch (int): number of epoch
            batch_size (int): batch size"""

        print("Training starts.")
        with tf.Session() as sess:
            sess.run(self.init)
            for epoch in range(num_epoch):
                training_data_perm = np.random.permutation(self.training_data)
                num_batches = len(self.training_data) // batch_size

                for iteration in range(num_batches):
                    X_batch = training_data_perm[iteration * batch_size:(iteration + 1) * batch_size]
                    sess.run(self.train, feed_dict={self.X: X_batch})

                train_loss = self.loss.eval(feed_dict={self.X: X_batch})
                print("epoch {} loss {}".format(epoch, train_loss))

            self.predictions = self.output_layer.eval(feed_dict={self.X: self.testing_data})
        print("Training done.")

    def evaluate_autoencoder(self):
        """Evaluate the autoencoder using self.testing_data

        Returns:
            MSE (list): list of the MSE of each dimension"""

        print("Evaluation starts.")

        N_test = len(self.testing_data)
        N_col = len(self.testing_data[0])
        MSE = np.array([0.] * N_col)
        for j in range(N_test):
            MSE += (self.predictions[j] - self.testing_data[j]) ** 2
        MSE = MSE / N_test

        print("Evaluation done.")
        return MSE

if __name__ == '__main__':
    # Retrieve data
    mat = sio.loadmat('observations3.mat')
    data = mat['observations']
    autoencoder = Autoencoder(data=data,
                              splitting_percentage=0.8,
                              lr=0.01
                              )
    autoencoder.train_autoencoder(num_epoch=5,
                                  batch_size=150
                                 )
    MSE = autoencoder.evaluate_autoencoder()
    print(MSE)
