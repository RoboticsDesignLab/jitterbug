import tensorflow as tf
import numpy as np
import scipy.io as sio


class Autoencoder():
    """An autoencoder to reduce the input dimension and control the jitterbug in a smaller state space."""

    def __init__(self,
                 data,
                 splitting_percentage=0.8,
                 lr=0.01,
                 sess=None
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
        num_hid1 = 8
        num_hid2 = 4
        num_outputs = num_inputs
        actf = tf.nn.relu
        self.X = tf.placeholder(tf.float32, shape=[None, num_inputs])
        initializer = tf.variance_scaling_initializer()

        w1 = tf.Variable(initializer([num_inputs, num_hid1]), dtype=tf.float32)
        w2 = tf.Variable(initializer([num_hid1, num_hid2]), dtype=tf.float32)
        w3 = tf.Variable(initializer([num_hid2, num_outputs]), dtype=tf.float32)

        b1 = tf.Variable(tf.zeros(num_hid1))
        b2 = tf.Variable(tf.zeros(num_hid2))
        b3 = tf.Variable(tf.zeros(num_outputs))

        self.hid_layer1 = actf(tf.matmul(self.X, w1) + b1)
        self.hid_layer2 = actf(tf.matmul(self.hid_layer1, w2) + b2)
        self.output_layer = actf(tf.matmul(self.hid_layer2, w3) + b3)

        self.loss = tf.reduce_mean(tf.square(self.output_layer - self.X))

        optimizer = tf.train.AdamOptimizer(lr)
        self.train = optimizer.minimize(self.loss)

        self.init_op = tf.global_variables_initializer()

        self.sess = sess

        if (self.sess == None):
            self.sess = tf.Session()

        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()

    def train_autoencoder(self,
                          num_epoch=5,
                          batch_size=150,
                          save_path=None
                          ):
        """Train the autoencoder usinf self.training_data.

        Args:
            num_epoch (int): number of epoch
            batch_size (int): batch size"""

        print("Training starts.")
        for epoch in range(num_epoch):
            training_data_perm = np.random.permutation(self.training_data)
            num_batches = len(self.training_data) // batch_size

            for iteration in range(num_batches):
                X_batch = training_data_perm[iteration * batch_size:(iteration + 1) * batch_size]
                self.sess.run(self.train, feed_dict={self.X: X_batch})

            train_loss = self.loss.eval(session=self.sess,feed_dict={self.X: X_batch})
            print("epoch {} loss {}".format(epoch, train_loss))

        print("Training done.")
        self.saver.save(self.sess, save_path)
        print("Model saved in path: %s" % save_path)

    def evaluate_autoencoder(self):
        """Evaluate the autoencoder using self.testing_data

        Returns:
            MSE (list): list of the MSE of each dimension"""

        print("Evaluation starts.")
        predictions = self.sess.run(self.output_layer, feed_dict={self.X:self.testing_data})
        N_test = len(self.testing_data)
        N_col = len(self.testing_data[0])
        MSE = np.array([0.] * N_col)
        for j in range(N_test):
            MSE += (predictions[j] - self.testing_data[j]) ** 2
        MSE = MSE / N_test

        print("Evaluation done.")
        return MSE

    def load_autoencoder(self, save_path):
        """Use the trained encoder saved in the file "saved_path" to reduce the input dimension

         Args:
             save_path (str): path where the file containing the autoencoder model is saved"""

        self.saver.restore(self.sess, save_path)

    def encode(self, obs):
        """Encode the observation to reduce its dimesion

        Args:
            obs (list): observation of the Jitterbug

        Returns;
            Reduced_obs (list): observation in the reduced dimension space"""
        reduced_obs = self.sess.run(self.hid_layer2, feed_dict={self.X: obs})
        return reduced_obs

if __name__ == '__main__':
    # Retrieve data
    mat = sio.loadmat('observations3.mat')
    data = mat['observations']
    with tf.Session() as sess:
        autoencoder = Autoencoder(data=data,
                                  splitting_percentage=0.8,
                                  lr=0.01,
                                  sess=sess
                                  )
        autoencoder.train_autoencoder(num_epoch=1,
                                      batch_size=150,
                                      save_path="./autoencoder_model.ckpt"
                                      )
        MSE = autoencoder.evaluate_autoencoder()
        print(MSE)
        
        autoencoder.load_autoencoder(save_path="./autoencoder_model.ckpt")
        obs = [data[10]]
        print(autoencoder.encode(obs))

