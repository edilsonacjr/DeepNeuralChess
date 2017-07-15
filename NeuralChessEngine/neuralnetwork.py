# neuralnetwork.py
#
# Neural network model to evaluate chess boards
# Written by Kyle McDonell
#
# CS 251
# Spring 2016

import tensorflow as tf
import numpy as np
import chess
import database
import chessengine
import time


# Neural network model to evaluate chess boards
class Model(object):

    def __init__(self, restore=None):

        # Input variable and label
        self.x = tf.placeholder(tf.float32, shape=[None, 373])
        self._y = tf.placeholder(tf.float32, shape=[None, 1])

        # Output
        self.y = self.buildNetwork(self.x)

        # Build the optimizer
        self.cost = tf.reduce_mean(tf.abs(tf.sub(self.y, self._y)))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
        # Operation to initialize the variables
        self.init = tf.initialize_all_variables()

        # Start session
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.Session(config=config)

        # Restore weights
        if restore is not None:
            self.saver.restore(self.sess, restore)
            print("Checkpoint", restore, "restored")
        # Or initialize them
        else:
            self.sess.run(self.init)
            print("Initialized variables")


    # Build the network structure
    def buildNetwork(self, x):

        # Weights and biases for the first hidden layer
        globalW = tf.Variable(tf.random_normal([37, 20], stddev=0.35),
                              name="globalW")
        globalB = tf.Variable(tf.random_normal([20], stddev=0.35),
                               name="globalB")
        pieceW = tf.Variable(tf.random_normal([208, 100], stddev=0.35),
                              name="pieceW")
        pieceB = tf.Variable(tf.random_normal([100], stddev=0.35),
                               name="pieceB")
        squareW = tf.Variable(tf.random_normal([128, 50], stddev=0.35),
                               name="squareW")
        squareB = tf.Variable(tf.random_normal([50], stddev=0.35),
                               name="squareB")

        # Weights and biases for the second hidden layer
        w2 = tf.Variable(tf.random_normal([170, 75], stddev=0.35),
                              name="W2")
        b2 = tf.Variable(tf.random_normal([75], stddev=0.35),
                              name="B2")

        # Weights and biases for the output layer
        w3 = tf.Variable(tf.random_normal([75, 1], stddev=0.35),
                         name="W3")
        b3 = tf.Variable(tf.random_normal([1], stddev=0.35),
                         name="B3")


        # Network operation

        # First hidden layer
        globalOutput = tf.add(tf.matmul(tf.slice(x, [0, 0], [-1, 37]), globalW), globalB)
        pieceOutput = tf.add(tf.matmul(tf.slice(x, [0, 37], [-1, 208]), pieceW), pieceB)
        squareOutput = tf.add(tf.matmul(tf.slice(x, [0, 37+208], [-1, -1]), squareW), squareB)
        layer1Output = tf.concat(1, [globalOutput, pieceOutput, squareOutput])

        # Second hidden layer
        layer2Input = tf.nn.relu6(layer1Output)
        layer2Output = tf.add(tf.matmul(layer2Input, w2), b2)

        # Output layer
        layer3Input = tf.nn.relu6(layer2Output)
        layer3Output = tf.add(tf.matmul(layer3Input, w3), b3)
        return tf.tanh(layer3Output)

    # Run an input through the network
    def runInput(self, feature):
        if len(feature.shape) == 1:
            feature = feature.reshape(1, -1)
        return self.sess.run(self.y, feed_dict={self.x: feature})

    # Return the model's session
    def getSession(self):
        return self.sess

    # Save the model weights
    def save(self, name=''):
        save_path = self.saver.save(self.sess, name)
        print("Model saved in file: %s" % save_path)

    # Bootstrap training using a labeled database of chess features
    def bootstrap(self, trainDB=None, testDB=None, batchSize=1000, epochs=50, displayStep=1):

        # Build the database if not provided
        if trainDB is None or testDB is None:
            trainDB = database.FeatDB('gameDB/trainX.npy', 'gameDB/sfTrainY.npy')
            testDB = database.FeatDB('gameDB/testX.npy', 'gameDB/sfTestY.npy')

        # Set up the training params
        dbSize = trainDB.size()
        iPerE = dbSize / batchSize
        # Start training
        iteration = 0
        epoch = 0
        while epoch < epochs:
            while iteration < iPerE:

                # Get the next batch
                batch_xs, batch_ys = trainDB.getNextBatch(batchSize)
                # Fit training using batch data
                self.sess.run(self.optimizer, feed_dict={self.x: batch_xs, self._y: batch_ys})

                # Calculate batch loss on display steps
                if epoch % displayStep == 0 and iteration == 0:
                    batchError = self.sess.run(self.cost, feed_dict={self.x: batch_xs,
                                                                self._y: batch_ys})
                    print("Epoch ", epoch, " - Minibatch Avg Error " + "{:.6f}".format(
                        batchError))

                iteration += 1

            epoch += 1
            iteration = 0

        print("Optimization Finished!")
        # Calculate accuracy for test set
        print("Testing Error:", self.sess.run(self.cost,
                                         feed_dict={self.x: testDB.getFeats(),
                                                    self._y: testDB.getLabels()}))




    # Calculates the accuracy for the test set
    def test(self, testDB=None):
        if testDB is None:
            testDB = database.FeatDB('gameDB/testX.npy', 'gameDB/sfTestY.npy')
        print("Testing Error:", \
            self.sess.run(self.cost, feed_dict={self.x: testDB.getFeats(),
                                                self._y: testDB.getLabels()}))



# Neural network chess engine
class NeuralNet(chessengine.ChessEngine):

    def __init__(self, depth=3, model=None):
        chessengine.ChessEngine.__init__(self, depth)
        self.model = model
        if model is None:
            self.model = Model('checkpoints/sfTrainingAdam3.ckpt')#td100its.ckpt')

    # Evaluate the board position using the neural network
    def evaluate(self, board):
        feat = database.getBoardFeature(board)
        output = self.model.runInput(feat)
        return output




model = Model()
model.bootstrap()
model.save('checkpoints/sfBoot.ckpt')

# model = Model(restore='checkpoints/td100its.ckpt')
# model.tdTrain(iterations=10)
# model.save('checkpoints/td110its.ckpt')

#model = Model(restore='checkpoints/td100its.ckpt')
#model.test()
