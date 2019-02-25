import numpy as np
from utils import load_data

import tensorflow as tf
import os
import matplotlib.pyplot as plt

#Recursive Neural Network for binary classification.
#It also has support to train a Recursive Neural Tensor Network.
#This is a naive implementation in tensorflow as for each sentence we create a different tree structure in tensorflow
#It takes a lot of memory, so we need to train the model in batches. On each batch we save the model on disk and load
#a new batch of sentences.

class RecursiveNN():
    def __init__(self, embeddings, word2idx,  l2_reg=0.001, use_rntn = False):
        np.random.seed(128)
        self.l2_reg = l2_reg
        self.use_rntn = use_rntn;
        # Embeddings dimentionality
        self.D = embeddings.shape[1];
        # Embeddings matrix
        self.embeddings = embeddings;
        self.word2idx = word2idx;
        self.train_operations = None;

    def fit(self, trees, n_epochs=50, checkpoint_file="./recurrent_nn.ckpt"):
        # Creates a graph for each one of the trees in the training set, trains it and saves the final model on disk.
        self.checkpoint_file = checkpoint_file;

        if (self.train_operations is None):
            self._build(trees)

        self.sess = tf.Session();

        if (os.path.isfile(checkpoint_file + ".index")):
            self.saver.restore(self.sess, checkpoint_file)
        else:
            self.sess.run(self.init)

        losses = []
        n_batch = len(self.train_operations)
        for epoch in range(n_epochs):
            batch_loss = 0
            batch_correct = 0
            for operation in self.train_operations:
                prediction, is_correct, loss, __ = self.sess.run(operation)
                batch_loss += loss
                batch_correct += is_correct[0]
            print("Epoch:", epoch, "Accuracy:", batch_correct / n_batch, "Loss:", batch_loss)
            losses.append(batch_loss)

        plt.plot(losses)
        plt.title("Train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        self.saver.save(self.sess, checkpoint_file)

    def train_score(self, checkpoint_file):

        if (self.train_operations is None):
            print("Training graph does not exist.")
            return;

        self.sess = tf.Session();
        try:
            self.saver.restore(self.sess, checkpoint_file)
            return self._calculate_score(self.train_operations)
        except Exception:
            print("File not found.....")

    def test_score(self, trees, checkpoint_file=None):

        if (checkpoint_file is None):
            checkpoint_file = self.checkpoint_file;

        if (self.train_operations is None):
            print("Training graph does not exist.")
            return;

        operations = self._compile_nodes(trees)

        self.sess = tf.Session();
        try:
            self.saver.restore(self.sess, checkpoint_file)
            return self._calculate_score(operations);
        except Exception:
            print("File not found.....")

    def _calculate_score(self, operations):

        total_correct = 0
        n_batch = len(operations)
        for operation in operations:
            is_correct = self.sess.run(operation[1])
            total_correct += is_correct[0]
        return total_correct / n_batch

    def _build(self, trees):
        self.W = tf.Variable(np.random.rand(2, self.D, self.D) / np.sqrt(self.D), dtype=tf.float32)
        self.bh = tf.Variable(np.zeros(self.D), dtype=tf.float32)

        if(self.use_rntn):
            self.All = tf.Variable(np.random.rand(self.D, self.D, self.D)*2/np.sqrt(self.D), dtype = tf.float32)
            self.Arr = tf.Variable(np.random.rand(self.D, self.D, self.D)*2/np.sqrt(self.D), dtype = tf.float32)
            self.Alr = tf.Variable(np.random.rand(self.D, self.D, self.D)*2/np.sqrt(self.D), dtype = tf.float32)

        self.activation = tf.nn.relu
        # Binary classifier. In case we decide to use all ratings change this value to the number of unique options.
        self.n_outputs = 2

        self.train_operations = self._compile_nodes(trees)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def _quadratic_multiplication(self, x, A, x2):
        #x is (1xD)
        #A is (DxD)
        #x2 is (1xD)
        result = tf.tensordot(tf.tensordot(x, A, axes=[-1, 0]), x2, axes = [-1, -1])
        return tf.reshape(result, (1, self.D))

    def _create_tensorflow_tree(self, node):
        # We create a graph in tensorflow followiing the same structure as the original tree
        if (node is None):
            return;
        left = self._create_tensorflow_tree(node.left)
        right = self._create_tensorflow_tree(node.right)
        if (node.word is not None):
            # If the node is a node word then return the corresponding word embeddings
            if (node.word in self.word2idx):
                word_idx = self.word2idx[node.word]
            else:
                word_idx = self.word2idx['unk']

            return self.embeddings[[word_idx]]  # Shape 1x50
        else:
            logits = tf.matmul(left, self.W[0]) + tf.matmul(right, self.W[1]) + self.bh

            if (self.use_rntn):
                logits += (self._quadratic_multiplication(left, self.All, left) +
                           self._quadratic_multiplication(left, self.Alr, right) +
                           self._quadratic_multiplication(right, self.Arr, right));
            return self.activation(logits)

    def _create_train_operations(self, tree):

        output = self._create_tensorflow_tree(tree)
        labels = [tree.rating]
        # A final layer is added to have two outputs and have a binary classifier.
        logits = tf.layers.dense(output, units=self.n_outputs, kernel_initializer=tf.variance_scaling_initializer,
                                 name="output_layer", reuse=tf.AUTO_REUSE)

        prediction = tf.argmax(logits, axis=-1)
        correct = tf.cast(tf.equal(prediction, labels), dtype=tf.int16)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        base_loss = tf.reduce_mean(xentropy)
        W_loss = self.l2_reg * tf.nn.l2_loss(self.W)
        total_loss = base_loss + W_loss

        # We want to reuse the momentum of each one of our trainable variables in the graph
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            opt_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss)

        return prediction, correct, total_loss, opt_step;

    def _compile_nodes(self, trees):
        operations = []
        for i, tree in enumerate(trees):
            prediction, correct, loss, opt_step = self._create_train_operations(tree)
            operations.append([prediction, correct, loss, opt_step])
            print("Tree", i, "created...")
        return operations;


# load in pre-trained word vectors


if __name__ == "__main__":
    embeddings, word2idx, train_sentences, test_sentences = load_data()

    tf.reset_default_graph()
    model = RecursiveNN(embeddings, word2idx)

    model.fit(train_sentences[:50])

    print("Train accuracy: ", model.train_score("./recurrent_nn.ckpt"))
    print("Test accuracy: ", model.test_score(test_sentences[:50]))
