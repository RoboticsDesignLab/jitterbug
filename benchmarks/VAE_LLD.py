import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import random
import scipy.io as sio

tfd = tfp.distributions


def KL_Normal(mu_1,
              log_sigma_1,
              mu_2,
              log_sigma_2,
              ):
    """Compute the KL divergence KL(p1,p2) with p1~N(mu_1,sigma_1)
    and p2~N(mu_2,sigma_2).

    Args:
    	mu_1 (Tensor): mean of N(mu_1,sigma_1)
    	log_sigma_sq_1 (Tensor): log(sigma_1^2) with sigma_1 being the standard deviation of N(mu_1,sigma_1)
    	mu_2 (Tensor): mean of N(mu_2,sigma_2)
    	log_sigma_sq_2 (Tensor): log(sigma_2^2) with sigma_2 being the standard deviation of N(mu_2,sigma_2)

    Returns:
    	KL(p1,p2)
    """
    sigma_1_sq = tf.exp(2 * log_sigma_1)
    sigma_2_sq = tf.clip_by_value(tf.exp(2 * log_sigma_2), 1e-15, np.float64('Inf'))
    return tf.reduce_mean(-0.5
                          + log_sigma_2
                          - log_sigma_1
                          + tf.divide(sigma_1_sq + tf.pow(mu_1 - mu_2, 2),
                                      2 * sigma_2_sq
                                      ),
                          1
                          )


def KL_Multivariate_Normal(mu_1,
                           cov_1,
                           mu_2,
                           cov_2,
                           k
                           ):
    """Compute the KL divergence KL(p1,p2) with p1~Mult_N(mu_1,cov_1)
    and p2~Mult_N(mu_2,cov_2).

    Args:
        mu_1 (Tensor): mean of Mult_N(mu_1,cov_1),
        cov_1 (Tensor): covariance matrix of Mult_N(mu_1,cov_1)
        mu_2 (Tensor): mean of Mult_N(mu_2,cov_2),
        cov_2 (Tensor): covariance matrix of Mult_N(mu_2,cov_2)
        k (int): dimension of the distributions

    Returns:
        KL(p1,p2)
    """
    tr = tf.linalg.trace(tf.matmul(tf.linalg.inv(cov_2),
                                   cov_1
                                   )
                         )
    means = tf.expand_dims(mu_2 - mu_1,
                           2
                           )
    m = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(means,
                                                    perm=[0, 2, 1]),
                                       tf.linalg.inv(cov_2)
                                       ),
                             means),
                   1
                   )

    det_2 = tf.clip_by_value(tf.linalg.det(cov_2),
                             1e-15,
                             np.float64('Inf')
                             )
    det_1 = tf.clip_by_value(tf.linalg.det(cov_1),
                             1e-15,
                             np.float64('Inf')
                             )
    log_det = tf.log(det_2) - tf.log(det_1)

    return 0.5 * tf.reduce_mean(tr + m - np.float64(k) + log_det,
                                1)


def log_normal(mu,
               log_sigma,
               x,
               ):
    """Compute log(p) with p~N(mu,sigma)

    Args:
        mu (Tensor): mean of N(mu,sigma)
        log_sigma (Tensor): log(sigma) with sigma being the standard deviation of N(mu,sigma)

    Returns:
        log(p)
    """
    sigma_sq = tf.clip_by_value(tf.exp(2 * log_sigma), 1e-15, np.float64('Inf'))
    p_x = tf.divide(tf.exp(-tf.divide(tf.pow(x - mu, 2),
                                      2 * sigma_sq)),
                    tf.pow(np.float64(2 * np.pi) * sigma_sq, 0.5)
                    )
    p_x = tf.clip_by_value(p_x, 1e-15, 1.)

    return tf.log(p_x)


def preprocess_data(obs,
                    act,
                    h=1,
                    batch_size=1000,
                    ):
    """Preprocess the data.
        Args:
            obs (numpy array): Numpy array of observations
            act (numpy array): Numpy array of actions
            h (int): Dynamics horizon
            batch_size (int): Batch size used to collect the data
        Returns:
            (numpy array): Concatenation of observations and actions split into batches of size batch_size-1
        """

    assert h >= 0, \
        "Dynamics horizon must be h >= 0, but was {}".format(h)

    # Apply dynamics horizon offset to segment training and testing data
    input = obs[0:-h, :].copy()
    target = obs[h:].copy()

    data = np.concatenate([input, target], axis=1)
    data = np.concatenate([data, act[0:-h]], axis=1)

    # Split the data into batches of size batch_size-1 while removing the last element of each batch
    N_batch = len(data) // batch_size
    data_batch = np.array([data[j * batch_size: (j + 1) * batch_size - 1, :] for j in range(N_batch + 1)])

    return data_batch


class VAE_LLD():
    """A variational autoencoder with linear latent space.
    """

    def __init__(self,
                 input_dim,
                 latent_dim,
                 act_dim,
                 act_fn=tf.nn.relu,
                 lr=1e-3,
                 ld=1e-1,
                 ):
        """
        input_dim (int): dimension of the input
        latent_dim (int): dimension of the latent space
        act_dim (int): dimension of the action space
        act_fn (tf.nn function): activation function
        lr (float): learning rate
        ld (float): weight lambda for contraction term in loss
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.act_fn = act_fn
        self.x_t = tf.placeholder(tf.float64, shape=[None, input_dim])
        self.x_t_1 = tf.placeholder(tf.float64, shape=[None, input_dim])
        self.u_t = tf.placeholder(tf.float64, shape=[None, act_dim])

        self.initializer = tf.random_normal_initializer(mean=0.0,
                                                        stddev=.1,
                                                        )
        zt_mu, zt_log_sigma, zt_1_mu, zt_1_log_sigma, xhat_t_mu, xhat_t_log_sigma, xhat_t_1_mu, xhat_t_1_log_sigma, A_t, B_t, o_t = self.build()

        # Compute loss
        # L_bound
        self.KL_Z = KL_Normal(zt_mu, zt_log_sigma, np.float64(0.), tf.log(np.float64(1.)))

        self.log_likelihoods = tf.reduce_mean(log_normal(xhat_t_mu, xhat_t_log_sigma, self.x_t)
                                              + log_normal(xhat_t_1_mu, xhat_t_1_log_sigma, self.x_t_1),
                                              1
                                              )

        self.L_bound = tf.reduce_mean(- self.log_likelihoods + self.KL_Z)

        # KL divergence Z_t_1
        mean_1 = tf.squeeze(tf.matmul(A_t,
                                      tf.expand_dims(zt_mu, 2))
                            + tf.matmul(B_t,
                                        tf.expand_dims(self.u_t, 1)),
                            2) + o_t

        cov_1 = tf.matmul(tf.matmul(A_t,
                                    tf.matrix_diag(tf.exp(2 * zt_1_log_sigma))
                                    ),
                          tf.transpose(A_t, perm=[0, 2, 1])
                          )

        cov_2 = tf.clip_by_value(tf.matrix_diag(tf.exp(2 * zt_1_log_sigma)),
                                 1e-15,
                                 np.float64('Inf')
                                 )

        self.KL_Z_1 = tf.reduce_mean(KL_Multivariate_Normal(mean_1,
                                                            cov_1,
                                                            zt_1_mu,
                                                            cov_2,
                                                            k=self.latent_dim
                                                            )
                                     )

        self.loss = self.L_bound + ld * self.KL_Z_1

        optimizer = tf.train.AdamOptimizer(lr)
        self.train = optimizer.minimize(self.loss)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def x2z(self,
            x,
            reuse,
            ):
        """Compute the latent variable z based on the input x.

        Args:
            x (Tensor): input
            reuse (bool): whether to reuse or not the network

        Returns:
            z (Tensor): latent variable
            z_mu (Tensor): mean of the z normal distribution
            z_log_sigma (Tensor): log(sigma), sigma being the standard deviation of the z normal distribution
        """
        enc1 = tf.layers.dense(x,
                               self.input_dim,
                               activation=self.act_fn,
                               name='enc1',
                               kernel_initializer=self.initializer,
                               reuse=reuse
                               )

        z_mu = tf.layers.dense(enc1,
                               self.latent_dim,
                               activation=None,
                               name='z_mu',
                               kernel_initializer=self.initializer,
                               reuse=reuse
                               )

        z_log_sigma = tf.layers.dense(enc1,
                                      self.latent_dim,
                                      activation=None,
                                      name="z_log_sigma",
                                      kernel_initializer=self.initializer,
                                      reuse=reuse,
                                      )

        z_eps = tf.random_normal(shape=tf.shape(z_log_sigma),
                                 mean=0,
                                 stddev=1,
                                 dtype=tf.float64
                                 )

        # Reparameterization trick
        z = z_mu + tf.exp(z_log_sigma) * z_eps
        return z, z_mu, z_log_sigma

    def z2x(self,
            z,
            reuse,
            ):
        """Compute the latent reconstruction of the input x based on the latent variable z.

        Args:
            z (Tensor): latent variable
            reuse (bool): whether to reuse or not the network

        Returns:
            x (Tensor): reconstruction of the input
            x_mu (Tensor): mean of the x normal distribution
            x_log_sigma (Tensor): log(sigma), sigma being the standard deviation of the x normal distribution
        """
        dec1 = tf.layers.dense(z,
                               self.input_dim,
                               activation=self.act_fn,
                               name='dec1',
                               kernel_initializer=self.initializer,
                               reuse=reuse
                               )

        x_mu = tf.layers.dense(dec1,
                               self.input_dim,
                               activation=None,
                               name='x_mu',
                               kernel_initializer=self.initializer,
                               reuse=reuse
                               )

        x_log_sigma = tf.layers.dense(dec1,
                                      self.input_dim,
                                      activation=None,
                                      name="x_log_sigma",
                                      kernel_initializer=self.initializer,
                                      reuse=reuse,
                                      )

        x_eps = tf.random_normal(shape=tf.shape(x_log_sigma),
                                 mean=0,
                                 stddev=1,
                                 dtype=tf.float64
                                 )

        # Reparameterization trick
        x = x_mu + tf.exp(x_log_sigma) * x_eps

        return x, x_mu, x_log_sigma

    def z2z(self,
            z,
            reuse=False,
            ):
        """Compute the latent variable z at timestep t+1 based on the latent variable z at timestep t.

               Args:
                   z (Tensor): latent variable at t
                   reuse (bool): whether to reuse or not the network

               Returns:
                   z_1 (Tensor): latent variable at t+1
                   A_t, B_t, o_t (Tensor): linearization coefficients
        """
        trans1 = tf.layers.dense(z,
                                 self.latent_dim,
                                 activation=self.act_fn,
                                 name="trans1",
                                 kernel_initializer=self.initializer,
                                 reuse=reuse,
                                 )

        v_t = tf.layers.dense(trans1,
                              self.latent_dim,
                              activation=None,
                              name="v_t",
                              kernel_initializer=self.initializer,
                              reuse=reuse,
                              )

        r_t = tf.layers.dense(trans1,
                              self.latent_dim,
                              activation=None,
                              name="r_t",
                              kernel_initializer=self.initializer,
                              reuse=reuse,
                              )

        B_t = tf.layers.dense(trans1,
                              self.latent_dim * self.act_dim,
                              activation=None,
                              name="B_t",
                              kernel_initializer=self.initializer,
                              reuse=reuse,
                              )

        o_t = tf.layers.dense(trans1,
                              self.latent_dim,
                              activation=None,
                              name="o_t",
                              kernel_initializer=self.initializer,
                              reuse=reuse,
                              )

        v_t = tf.expand_dims(v_t, -1)
        r_t_T = tf.expand_dims(r_t, 1)
        A_t = tf.eye(self.latent_dim, dtype=tf.float64) + tf.matmul(v_t, r_t_T)

        B_t = tf.reshape(B_t,
                         np.array([-1, self.latent_dim, self.act_dim],
                                  dtype="int32"))

        z_1 = tf.squeeze(tf.matmul(A_t,
                                   tf.expand_dims(self.z_t, 2))
                         + tf.matmul(B_t,
                                     tf.expand_dims(self.u_t, 1)),
                         2) + o_t

        return z_1, A_t, B_t, o_t

    def build(self):
        """Build the network."""
        # x_t -> z_t
        self.z_t, zt_mu, zt_log_sigma = self.x2z(self.x_t,
                                                 reuse=False,
                                                 )
        # x_t_1 -> z_t_1
        self.z_t_1, zt_1_mu, zt_1_log_sigma = self.x2z(self.x_t_1,
                                                       reuse=True,
                                                       )

        # z_t -> x_hat_t
        self.x_hat_t, xhat_t_mu, xhat_t_log_sigma = self.z2x(self.z_t,
                                                             reuse=False,
                                                             )

        # z_t -> z_hat_t_1
        self.z_hat_t_1, A_t, B_t, o_t = self.z2z(self.z_t,
                                                 reuse=False,
                                                 )

        # z_hat_t_1 -> x_hat_t_1
        self.x_hat_t_1, xhat_t_1_mu, xhat_t_1_log_sigma = self.z2x(self.z_hat_t_1,
                                                                   reuse=True,
                                                                   )

        return zt_mu, zt_log_sigma, zt_1_mu, zt_1_log_sigma, xhat_t_mu, xhat_t_log_sigma, xhat_t_1_mu, xhat_t_1_log_sigma, A_t, B_t, o_t

    def encode(self, x):
        return self.sess.run(self.z_t, feed_dict={self.x_t: x})

    def decode(self, z):
        return self.sess.run(self.x_hat_t, feed_dict={self.z_t: z})

    def autoencode(self, x):
        return self.sess.run(self.x_hat_t, feed_dict={self.x_t: x})

    def train_autoencoder(self,
                          training_data,
                          num_epoch=5,
                          save_path=None
                          ):
        """Train the autoencoder using training_data.

        Args:
            training_data (array or list): data used to train the autoencoder
            num_epoch (int): number of epoch
            save_path (str): path where to save the autoencoder once trained
        """
        print("Training starts.")
        num_batches = len(training_data)
        index_list = np.array(range(num_batches))
        for epoch in range(num_epoch):
            # Shuffle the batches
            random.shuffle(index_list)
            for iteration in index_list:
                X_batch = training_data[iteration]
                # Shuffle the elements of each batch
                random.shuffle(X_batch)
                self.sess.run(self.train,
                              feed_dict={self.x_t: X_batch[:, :self.input_dim],
                                         self.x_t_1: X_batch[:, self.input_dim:2 * self.input_dim],
                                         self.u_t: np.transpose([X_batch[:, -1]])
                                         }
                              )
            L_bound, KL_Z, loss = self.sess.run(
                [self.L_bound, self.KL_Z_1, self.loss],
                feed_dict={self.x_t: X_batch[:, :self.input_dim],
                           self.x_t_1: X_batch[:, self.input_dim:2 * self.input_dim],
                           self.u_t: np.transpose([X_batch[:, -1]])
                           }
            )

            print("epoch {}: L_bound {} | KL_Z {} | loss {} ".format(epoch, L_bound, KL_Z, loss))

        print("Training done.")
        if save_path != None:
            self.save_autoencoder(save_path)

    def evaluate_autoencoder(self,
                             testing_data):
        """Evaluate the autoencoder using self.testing_data

                Args:
                    testing_data (array or list): data used to evaluate the performance of the autoencoder
                Returns:
                    MSE (list): list of the MSE of each dimension"""

        print("Evaluation starts.")
        N_test = len(testing_data)
        batch_size = len(testing_data[0])

        MSE = np.array([[0.] * self.input_dim])
        for i in range(N_test):
            test_batch = testing_data[i]
            predictions = self.autoencode(test_batch[:, :self.input_dim])

            for j in range(batch_size):
                MSE += (predictions[j] - test_batch[j, :self.input_dim]) ** 2
        MSE = MSE / (N_test * batch_size)

        print("Evaluation done.")
        return MSE

    def save_autoencoder(self):
        pass


if __name__ == "__main__":
    # Retrieve data
    mat_obs = sio.loadmat('observations_mid_random_normalized.mat')
    data_obs = mat_obs['observations']

    mat_act = sio.loadmat('actions_mid_random.mat')
    data_act = mat_act['observations']

    data = preprocess_data(data_obs,
                           data_act,
                           )

    # Split the data into training and testing sets
    N = len(data)
    splitting_percentage = 0.8
    splitting_int = int(round(splitting_percentage * N, 0))
    training_data = data[:splitting_int]
    testing_data = data[splitting_int:]

    # Create the VAE
    VAE = VAE_LLD(input_dim=len(data_obs[0]),
                  latent_dim=15,
                  act_dim=len(data_act[0]),
                  lr=1e-4,
                  ld=1e-1,
                  )

    VAE.train_autoencoder(training_data=training_data,
                          num_epoch=100)
    MSE = VAE.evaluate_autoencoder(testing_data=testing_data)
    datapoint = testing_data[500][600, :19]

    print(MSE)
