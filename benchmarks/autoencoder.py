import os

# Uncomment to disable GPU training in tensorflow (must be before keras imports)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import scipy.io as sio
import pickle


def normalize(data,fileName):
    """Normalize the features af the dataset so that they vary between -1 and 1 and save them in a new Matlab file.
    Save the extrema in a pickle file.

    Args:
        data (array or list): dataset
        fileName (str): name of the Matlab file (without the .mat extension) that will contain the normalized data"""
    print("Normalizing data.")
    extremaFile = open(fileName+"_extrema.pkl","wb")
    N_lines = len(data)
    N_cols = len(data[0])
    normalized_data = np.array([[0.]*N_cols]*N_lines)
    extrema = []
    for col in range(N_cols):
        Xmin = float('Inf')
        Xmax = -float('Inf')
        for line in range(N_lines):
            val = data[line][col]
            if val>Xmax:
                Xmax=val
            if val<Xmin:
                Xmin=val
        extrema.append([Xmin,Xmax])

        for line2 in range(N_lines): #normalize between -1 and 1
            normalized_data[line2][col] = (data[line2][col]-Xmin)/(Xmax-Xmin)
    print(extrema)

    pickle.dump(extrema,extremaFile)
    sio.savemat(fileName+".mat", mdict={'observations': normalized_data})
    extremaFile.close()
    print("Data normalized and saved in file "+fileName+".mat")



class Autoencoder():
    """An autoencoder to reduce the input dimension and control the jitterbug in a smaller state space."""

    def __init__(self,
                 feature_dimension,
                 lr=0.01,
                 sess=None
                 ):

        self.lr = lr

        # Build the autoencoder
        num_inputs = feature_dimension
        num_hid1 = 12
        #num_hid2 = 16
        #num_hid3 = 14
        #num_hid4 = 12
        #num_hid5 = 14
        #num_hid6 = 16
        #num_hid7 = 16
        num_outputs = num_inputs
        #actf = tf.nn.relu
        #actf = tf.keras.activations.linear
        actf = tf.nn.tanh
        self.X = tf.placeholder(tf.float32, shape=[None, num_inputs])
        #initializer = tf.variance_scaling_initializer()
        initializer = tf.random_normal_initializer(mean=0.0,
                                                   stddev=1,
                                                   )

        w1 = tf.Variable(initializer([num_inputs, num_hid1]), dtype=tf.float32)
        w2 = tf.transpose(w1)
        #w3 = tf.Variable(initializer([num_hid2, num_hid3]), dtype=tf.float32)
        #w4 = tf.Variable(initializer([num_hid3, num_hid4]), dtype=tf.float32)
        #w5 = tf.transpose(w4)
        #w6 = tf.transpose(w3)
        #w7 = tf.transpose(w2)
        #w8 = tf.transpose(w1)

        #initializer_constant = tf.constant(np.identity(num_inputs), dtype=tf.float32)
        #w1 = tf.get_variable("w1", initializer=initializer_constant, dtype=tf.float32)
        #w2 = tf.get_variable("w2", initializer=initializer_constant, dtype=tf.float32)

        b1 = tf.Variable(tf.zeros(num_hid1))
        b2 = tf.Variable(tf.zeros(num_outputs))
        #b3 = tf.Variable(tf.zeros(num_hid3))
        #b4 = tf.Variable(tf.zeros(num_hid4))
        #b5 = tf.Variable(tf.zeros(num_hid5))
        #b6 = tf.Variable(tf.zeros(num_hid6))
        #b7 = tf.Variable(tf.zeros(num_hid7))
        #b8 = tf.Variable(tf.zeros(num_outputs))

        self.hid_layer1 = actf(tf.matmul(self.X, w1) + b1)
        #self.hid_layer2 = actf(tf.matmul(self.hid_layer1, w2) + b2)
        #self.hid_layer3 = actf(tf.matmul(self.hid_layer2, w3) + b3)
        #self.hid_layer4 = actf(tf.matmul(self.hid_layer3, w4) + b4)
        #self.hid_layer5 = actf(tf.matmul(self.hid_layer4, w5) + b5)
        #self.hid_layer6 = actf(tf.matmul(self.hid_layer5, w6) + b6)
        #self.hid_layer7 = actf(tf.matmul(self.hid_layer6, w7) + b7)
        self.output_layer = actf(tf.matmul(self.hid_layer1, w2) + b2)

        #self.hid_layer1 = tf.matmul(self.X, w1) + b1
        #self.output_layer = tf.matmul(self.hid_layer1, w2) + b2
        self.loss = tf.reduce_mean(tf.square(self.output_layer - self.X))

        optimizer = tf.train.AdamOptimizer(lr)
        self.train = optimizer.minimize(self.loss)

        self.init_op = tf.global_variables_initializer()

        self.sess = sess

        if (self.sess == None):
            self.sess = tf.Session()

        self.sess.run(self.init_op)
        self.saver = tf.train.Saver()

        #File that contains the extrema to normalize the observations
        #fileName = './observations4_move_in_direction_normalized_extrema.pkl'
        fileName = './observations5_use_policy_normalized_extrema.pkl'
        extrema = []
        with (open(fileName, "rb")) as openfile:
            while True:
                try:
                    extrema.append(pickle.load(openfile))
                except EOFError:
                    break
        self.extrema = extrema[0]

    def train_autoencoder(self,
                          training_data,
                          num_epoch=5,
                          batch_size=150,
                          save_path=None
                          ):
        """Train the autoencoder using self.training_data.

        Args:
            training_data (array or list): data used to train the autoencoder
            num_epoch (int): number of epoch
            batch_size (int): batch size
            save_path (str): path where to save the autoencoder once trained"""

        print("Training starts.")
        for epoch in range(num_epoch):
            training_data_perm = np.random.permutation(training_data)
            num_batches = len(training_data) // batch_size

            for iteration in range(num_batches):
                X_batch = training_data_perm[iteration * batch_size:(iteration + 1) * batch_size]
                self.sess.run(self.train, feed_dict={self.X: X_batch})

            train_loss = self.loss.eval(session=self.sess,feed_dict={self.X: X_batch})
            print("epoch {} loss {}".format(epoch, train_loss))

        print("Training done.")
        if save_path != None:
            self.save_autoencoder(save_path)

    def train_using_batch(self,
                          counter,
                          batch,
                          ):
        """Train the autoencoder using only one batch.

        Args:
            counter (int): number of time the function has been called, i.e. number of epoch
            batch (array): batch used to train the autoencoder"""

        self.sess.run(self.train, feed_dict={self.X: batch})
        train_loss = self.loss.eval(session=self.sess, feed_dict={self.X: batch})
        print("epoch {} loss {}".format(counter, train_loss))


    def evaluate_autoencoder(self, testing_data):
        """Evaluate the autoencoder using self.testing_data

        Args:
            testing_data (array or list): data used to evaluate the performance of the autoencoder
        Returns:
            MSE (list): list of the MSE of each dimension"""

        print("Evaluation starts.")
        predictions = self.sess.run(self.output_layer, feed_dict={self.X:testing_data})
        N_test = len(testing_data)
        N_col = len(testing_data[0])
        MSE = np.array([0.] * N_col)
        for j in range(N_test):
            MSE += (predictions[j] - testing_data[j]) ** 2
        MSE = MSE / N_test

        print("Evaluation done.")
        return MSE


    def load_autoencoder(self, save_path):
        """Use the trained encoder saved in the file "saved_path" to reduce the input dimension

         Args:
             save_path (str): path where the file containing the autoencoder model is saved"""

        self.saver.restore(self.sess, save_path)

    def encode(self, obs):
        """Encode the observation to reduce its dimension

        Args:
            obs (list): observation of the Jitterbug

        Returns:
            reduced_obs (list): observation in the reduced dimension space"""
        reduced_obs = self.sess.run(self.hid_layer1, feed_dict={self.X: obs})
        #reduced_obs = self.hid_layer1.eval(session=sess,feed_dict={self.X: obs})
        return reduced_obs

    def decode(self, reduced_obs):
        """Decode an observation

        Args:
            reduced_obs (list): observation of the Jitterbug in the reduced dimension space

        Returns:
            obs (list): observation in the original dimension space"""
        obs = self.sess.run(self.output_layer, feed_dict={self.hid_layer1: reduced_obs})
        return obs

    def normalize_obs(self,obs):
        """Normalize the observation so that its features vary between -1 and 1

        Args:
            obs (array): observation to normalize

        Returns:
            normalized_observation (array): normalized observation"""
        N_col = len(self.extrema)
        normalized_obs = [0.]*N_col
        for col in range(N_col):
            Xmin, Xmax = self.extrema[col]
            normalized_obs[col]=-1 + 2 * (obs[col] - Xmin) / (Xmax - Xmin)

        return normalized_obs

    def normalize_obs01(self,obs):
        """Normalize the observation so that its features vary between 0 and 1

               Args:
                   obs (array): observation to normalize

               Returns:
                   normalized_observation (array): normalized observation"""
        N_col = len(self.extrema)
        normalized_obs = [0.] * N_col
        for col in range(N_col):
            Xmin, Xmax = self.extrema[col]
            normalized_obs[col] = (obs[col] - Xmin) / (Xmax - Xmin)

        return normalized_obs

    def save_autoencoder(self, save_path):
        """Save the autoencoder model.

        Args:
            save_path (str): path where to save the autoencoder"""
        self.saver.save(self.sess, save_path)
        print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    # Retrieve data
    mat = sio.loadmat('observations3_random_normalized01.mat')
    data = mat['observations']

    mat2 = sio.loadmat('observations4_move_in_direction_normalized01.mat')
    data2 = mat2['observations']

    #mat3 = sio.loadmat('observations5_use_policy_normalized.mat')
    #data3 = mat3['observations']

    data_conc = np.concatenate((data,data2),axis=0)
    #print(data_conc)
    np.random.shuffle(data_conc)
    #print(data_conc)
    #print(data)
    N = len(data_conc)
    print(N)
    splitting_percentage = 0.7

    # Split the data into training and testing sets
    splitting_int = int(round(splitting_percentage * N, 0))
    training_data = data_conc[:splitting_int]
    testing_data = data_conc[splitting_int:]

    #normalize(data,'observations3_random_normalized')

    with tf.Session() as sess:
        #Autoencoder saved: 1, 2, 3, 4, 5
        autoencoder = Autoencoder(feature_dimension=len(data[0]),
                                  lr=0.001,
                                  sess=sess
                                  )
        #autoencoder.normalize_obs()
        obs = [data[10]]
        #print(data[10])
        autoencoder.train_autoencoder(training_data=training_data,
                                      num_epoch=200,
                                      batch_size=65536,
                                      save_path="./autoencoder_model000.ckpt"
                                      )

        #autoencoder.load_autoencoder(save_path="./autoencoder_model.ckpt")

        MSE = autoencoder.evaluate_autoencoder(testing_data=testing_data)
        print(MSE)


        #for i in range(1, 6):
            # Generate a session for each autoencoder
            #g_i = tf.Graph()
            #with g_i.as_default():
            #    session_i = tf.Session(graph=g_i)
            #    session_list.append(session_i)

        print("original data:")
        print(obs)
        reduced_obs=autoencoder.encode(obs)
        #print(reduced_obs)
        autoencoded_obs = autoencoder.decode(reduced_obs)
        print('autoencoded data:')
        print(autoencoded_obs)




