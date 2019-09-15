"""Train a Denoising Dynamics Auto Encoder"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
import scipy.io as sio

# Disable tensorflow GPU usage
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import tensorflow.contrib.slim as slim


class AutoEncoder(tf.estimator.Estimator):
    """A Denoising Auto Encoder Class

    Based on https://github.com/sebp/tf_autoencoder
    """

    def __init__(
        self,
        hidden_units,
        activation_fn=tf.nn.relu,
        dropout=None,
        weight_decay=1e-5,
        learning_rate=0.001,
        model_dir=None,
        config=None
    ):

        def _model_fn(features, labels, mode):
            """Define estimator architecture

            Args:
                features ():
                labels ():
                mode ():

            Returns:
                ():
            """
            return AutoEncoder._create_estimator_spec_from_logits(
                labels=labels,
                logits=AutoEncoder._fully_connected_autoencoder(
                    inputs=features,
                    hidden_units=hidden_units,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    weight_decay=weight_decay,
                    mode=mode
                ),
                learning_rate=learning_rate,
                mode=mode
            )

        super(AutoEncoder, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config
        )

    @staticmethod
    def _add_hidden_layer_summary(value):
        tf.summary.scalar('fraction_of_zero_values', tf.nn.zero_fraction(value))
        tf.summary.histogram('activation', value)

    @staticmethod
    def _fc_encoder(inputs, hidden_units, dropout, scope=None):
        net = inputs
        with tf.variable_scope(scope, 'encoder', [inputs]):
            tf.assert_rank(inputs, 2)
            for layer_id, num_hidden_units in enumerate(hidden_units):
                with tf.variable_scope(
                        'layer_{}'.format(layer_id),
                        values=(net,)) as layer_scope:
                    net = tf.contrib.layers.fully_connected(
                        net,
                        num_outputs=num_hidden_units,
                        scope=layer_scope)
                    if dropout is not None:
                        net = slim.dropout(net)
                    AutoEncoder._add_hidden_layer_summary(net)
            net = tf.identity(net, name='output')

        return net

    @staticmethod
    def _fc_decoder(inputs, hidden_units, dropout, scope=None):
        net = inputs
        with tf.variable_scope(scope, 'decoder', [inputs]):
            for layer_id, num_hidden_units in enumerate(hidden_units[:-1]):
                with tf.variable_scope(
                        'layer_{}'.format(layer_id),
                        values=(net,)) as layer_scope:
                    net = tf.contrib.layers.fully_connected(
                        net,
                        num_outputs=num_hidden_units,
                        scope=layer_scope)
                    if dropout is not None:
                        net = slim.dropout(net, scope=layer_scope)
                    AutoEncoder._add_hidden_layer_summary(net)

            with tf.variable_scope(
                    'layer_{}'.format(len(hidden_units) - 1),
                    values=(net,)) as layer_scope:
                net = tf.contrib.layers.fully_connected(net, hidden_units[-1],
                                                        activation_fn=None,
                                                        scope=layer_scope)
                tf.summary.histogram('activation', net)
            net = tf.identity(net, name='output')
        return net

    @staticmethod
    def _autoencoder_arg_scope(activation_fn, dropout, weight_decay, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        if weight_decay is None or weight_decay <= 0:
            weights_regularizer = None
        else:
            weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

        with slim.arg_scope(
                [tf.contrib.layers.fully_connected],
                weights_initializer=slim.initializers.variance_scaling_initializer(),
                weights_regularizer=weights_regularizer,
                activation_fn=activation_fn
        ), \
             slim.arg_scope(
                 [slim.dropout],
                 keep_prob=dropout,
                 is_training=is_training
             ) as arg_sc:
            return arg_sc

    @staticmethod
    def _fully_connected_autoencoder(
            inputs,
            hidden_units,
            activation_fn,
            dropout,
            weight_decay,
            mode,
            scope=None
    ):
        """Create autoencoder with fully connected layers.
        Parameters
        ----------
        inputs : tf.Tensor
            Tensor holding the input data.
        hidden_units : list of int
            Number of units in each hidden layer.
        activation_fn : callable|None
            Activation function to use.
        dropout : float|None
             Percentage of nodes to remain activate in each layer,
             or `None` to disable dropout.
        weight_decay : float|None
            Amount of regularization to use on the weights
            (excludes biases).
        mode : tf.estimator.ModeKeys
            The mode of the model.
        scope : str
            Name to use in Tensor board.
        Returns
        -------
        net : tf.Tensor
            Output of the decoder's reconstruction layer.
        """
        with tf.variable_scope(scope, 'FCAutoEnc', [inputs]):
            with slim.arg_scope(AutoEncoder._autoencoder_arg_scope(
                    activation_fn,
                    dropout,
                    weight_decay,
                    mode
            )
            ):
                net = AutoEncoder._fc_encoder(inputs, hidden_units, dropout)
                n_features = inputs.shape[1].value
                decoder_units = hidden_units[:-1][::-1] + [n_features]
                net = AutoEncoder._fc_decoder(net, decoder_units, dropout)

        return net

    @staticmethod
    def _create_estimator_spec_from_logits(labels, logits, learning_rate, mode):
        """Add loss function and create estimator spec.
        Parameters
        ----------
        labels : tf.Tensor
            Tenor holding the data to reconstruct.
        logits : tf.Tensor
            Tenor holding the reconstructed data.
        learning_rate : float
            Learning rate.
        mode : tf.estimator.ModeKeys
            The mode of the model.
        Returns
        -------
        spec : tf.estimator.EstimatorSpec
            Specification of the model.
        """
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        probs = tf.nn.sigmoid(logits)

        predictions = {"prediction": probs}
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

        tf.losses.sigmoid_cross_entropy(labels, logits)
        total_loss = tf.losses.get_total_loss(
            add_regularization_losses=is_training)

        train_op = None
        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=total_loss,
                optimizer="Adam",
                learning_rate=learning_rate,
                learning_rate_decay_fn=lambda lr,
                                              gs: tf.train.exponential_decay(
                    lr,
                    gs,
                    1000,
                    0.96,
                    staircase=True
                ),
                global_step=tf.train.get_global_step(),
                summaries=["learning_rate", "global_gradient_norm"])

            # Add histograms for trainable variables
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(
                    tf.cast(labels, tf.float64), tf.cast(probs, tf.float64))
            }

        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=total_loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )


def segment_data(data, h):
    """Segment input and target data for DDAE

    Args:
        data (numpy array): Nx16 Numpy array of input data
        h (int): Dynamics horizon, 0 corresponds to a regular denoising AE

    Returns:
        (numpy array): Input data
        (numpy array): Target data
    """

    assert h >= 0,\
        "Dynamics horizon must be h >= 0, but was {}".format(h)

    # Apply dynamics horizon offset to segment training and testing data
    if h == 0:
        input = data.copy()
        target = data.copy()
    else:
        input = data[0:-h, :].copy()
        target = data[h:].copy()

    return input, target


def main(
        *,
        l=[16, 12, 8, 4],
        h=0,
        batch_size=256,
        num_epochs=50000,
        **kwargs
):
    """Train a Denoising Dynamics AutoEncoder

    We use random data sampled from the Jitterbug face_direction task, which
    gives 16 dimensional observations.
    
    Args:
        l (int): Latent space dimension
        h (int): Dynamics horizon, 0 corresponds to a regular denoising AE
        batch_size (int): batch size

    """

    tf.logging.set_verbosity(tf.compat.v1.logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    print("Training DDAE with L = {}, H = {}".format(
        l,
        h
    ))

    # Retrieve data
    print("Preparing Data...")
    data = sio.loadmat(
        os.path.join(
            "observations3_random.mat"
        )
    )['observations']

    # Train, test split
    num_observations = len(data)
    num_train = int(num_observations * 0.7)

    # Convert data to tensorflow Dataset object
    def get_train_dataset():
        # Convert to dataset
        train_data = segment_data(data[:num_train], h)
        dataset = tf.data.Dataset.from_tensor_slices(train_data)
        return dataset.shuffle(100).repeat(num_epochs).batch(batch_size)

    def get_test_dataset():
        # Convert to dataset
        test_data = segment_data(data[num_train:], h)
        dataset = tf.data.Dataset.from_tensor_slices(test_data)
        return dataset.batch(batch_size)

    # Instantiate DAE with appropriate latent space
    print("Building AE...")
    mdl = AutoEncoder(
        hidden_units=l,
        **kwargs
    )

    # Train
    print("Training...")
    mdl.train(
        input_fn=get_train_dataset,
        steps=num_epochs
    )

    # Evaluate
    print("Evaluating...")
    print(
        mdl.evaluate(
            input_fn=get_test_dataset,
            steps=1
        )
    )


if __name__ == '__main__':

    import os
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--enc_layers",
        type=int,
        nargs='+',
        required=True,
        help="Encoder hidden layer sizes"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        required=False,
        help="Dynamics horizon"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Model path"
    )
    args = parser.parse_args()
    args.enc_layers = list(args.enc_layers)

    main(
        l=args.enc_layers,
        h=args.horizon,
        model_dir=args.model_dir
    )
