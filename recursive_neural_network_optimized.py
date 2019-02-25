import numpy as np
from utils import load_data, transform_tree_set_to_list

import tensorflow as tf
import os
import matplotlib.pyplot as plt

#Optimized version of the Recursive Neural Network for binary classification.
#It also has support to train a Recursive Neural Tensor Network.
#Instead of constructing a tensorflow tree for each sentence, we first convert the original tree into 4 lists
#that we will use later in tensorflow to fill in a symbolic matrix that will serve as a cache to store intermediate
#computations.

class RecursiveNN():
    def __init__(self, embeddings, l2_reg=0.001, learning_rate=0.01, use_rntn=False):
        # np.random.seed(128)
        self.learning_rate = learning_rate;
        self.l2_reg = l2_reg
        self.use_rntn = use_rntn;
        # Embeddings dimentionality
        self.D = embeddings.shape[1];
        # Embeddings matrix
        self.embeddings = embeddings;

    def fit(self, X, n_epochs=5):
        # Creates a graph for each one of the trees in the training set, trains it and saves the final model on disk.

        self._build()

        self.sess = tf.Session();
        self.sess.run(self.init)

        losses = []
        n_batch = len(X)
        for epoch in range(n_epochs):
            # np.random.shuffle(X)
            batch_loss = 0
            batch_correct = 0
            for i, x in enumerate(X):
                logits, prediction, is_correct, loss, __ = self.sess.run(
                    [self.logits, self.prediction, self.correct, self.total_loss, self.opt_step],
                    feed_dict={self.x_input: x})
                # print(logits, prediction, is_correct)
                batch_loss += loss
                batch_correct += is_correct[0]
            print("Epoch:", epoch, "Accuracy:", batch_correct / n_batch, "Loss:", batch_loss)
            losses.append(batch_loss)
        # self.stacked_eval = stacked;
        plt.plot(losses)
        plt.title("Train loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

        # self.saver.save(self.sess, checkpoint_file)

    def _build(self):
        self.x_input = tf.placeholder(dtype=tf.int32, shape=(None, 4))
        T = tf.shape(self.x_input)[0]
        self.W = tf.Variable(np.random.rand(2, self.D, self.D) / np.sqrt(self.D), dtype=tf.float32)
        self.bh = tf.Variable(np.zeros(self.D), dtype=tf.float32)
        if (self.use_rntn):
            self.All = tf.Variable(np.random.rand(self.D, self.D, self.D) * 2 / np.sqrt(self.D), dtype=tf.float32)
            self.Arr = tf.Variable(np.random.rand(self.D, self.D, self.D) * 2 / np.sqrt(self.D), dtype=tf.float32)
            self.Alr = tf.Variable(np.random.rand(self.D, self.D, self.D) * 2 / np.sqrt(self.D), dtype=tf.float32)
        self.hidden = tf.TensorArray(dtype=tf.float32, size=T)
        # I use the tanh because with relu the final layer will output a very large number
        self.activation = tf.nn.tanh
        # Binary classifier. In case we decide to use all ratings change this value to the number of unique options.
        self.n_outputs = 2

        def quad_mult(x, A, x2):
            # Quadratic multiplication
            # x is (D, )
            # A is (DxDxD)
            # x2 is (D, )
            result = tf.tensordot(tf.tensordot(x, A, axes=[0, 0]), x2, axes=[-1, 0])
            return result

        def linear_mult(x, W):
            return tf.tensordot(x, W, axes=[0, 0])

        def get_embedding(i, a):
            output = a.write(i, tf.nn.embedding_lookup(self.embeddings, self.x_input[i, 2]))
            return output;

        def update_node(i, a):
            left_child = a.read(self.x_input[i, 0])
            right_child = a.read(self.x_input[i, 1])
            logits_ = linear_mult(left_child, self.W[0]) + linear_mult(right_child, self.W[1]) + self.bh
            if (self.use_rntn):
                logits_ += (quad_mult(left_child, self.All, left_child) +
                            quad_mult(left_child, self.Alr, right_child) +
                            quad_mult(right_child, self.Arr, right_child));
            output = a.write(i, self.activation(logits_))
            return output;

        # [left, right, words, ratings]
        def body(i, a):
            node_op = tf.cond(tf.not_equal(self.x_input[i, 2], -1),
                              lambda: get_embedding(i, a),
                              lambda: update_node(i, a));
            return i + 1, node_op;

        n, final_hidden = tf.while_loop(lambda i, _: i < T, body, (0, self.hidden))

        # self.stacked_hidden = final_hidden.stack();

        output = tf.reshape(final_hidden.read(T - 1), (1, self.D));
        labels = [self.x_input[-1, 3]]
        # A final layer is added to have two outputs and have a binary classifier.
        logits = tf.layers.dense(output, units=self.n_outputs, kernel_initializer=tf.variance_scaling_initializer,
                                 name="output_layer")

        prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
        correct = tf.cast(tf.equal(prediction, labels), dtype=tf.int16)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        base_loss = tf.reduce_mean(xentropy)
        W_loss = self.l2_reg * tf.nn.l2_loss(self.W)
        total_loss = base_loss + W_loss

        opt_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.prediction = prediction;
        self.logits = logits;
        # self.output = output;
        self.correct = correct;
        self.total_loss = total_loss;
        self.opt_step = opt_step;

    def score(self, X):
        #Calculates accuracy
        correct = 0
        total = len(X)
        for i, x in enumerate(X):
            is_correct = self.sess.run(self.correct, feed_dict={self.x_input: x})
            correct += is_correct[0]
        return correct / total;




def main():

    #Load the sentences into a tree format
    embeddings, word2idx, train_sentences, test_sentences = load_data()

    #Convert each tree into four lists: Left child idx, Right child idx, Word embedding id, Rating.
    X_train = transform_tree_set_to_list(word2idx, train_sentences)
    X_test = transform_tree_set_to_list(word2idx, test_sentences)

    tf.reset_default_graph()
    model = RecursiveNN(embeddings, use_rntn=True, l2_reg=0.001, learning_rate=0.001)
    model.fit(X_train[:100], 10)

    print("Train accuracy: ", model.score(X_train[:100]))
    print("Test accuracy: ", model.score(X_test[:100]))

if __name__ == "__main__":
    main();