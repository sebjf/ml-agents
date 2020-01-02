import numpy as np
import gym
import random
import copy
from timeit import default_timer as timer

class Individual:
    def __init__(self, max_steps = 1000, weights = None):
        self.step_callback = None
        self.max_steps = max_steps
        self.weights = weights
        if self.weights is None:
            self.create_weights()

    def create_weights(self):
        model = Individual.create_model()
        self.weights = model.get_weights()

    @staticmethod
    def create_model():
        import keras
        keras.backend.clear_session()
        inputs = keras.Input(shape=(44,), name="vector_observation") # match the input in TensorNames.cs
        x = keras.layers.Dense(132, activation='relu')(inputs)
        x = keras.layers.Dense(132, activation='relu')(inputs)
        x = keras.layers.Dense(2, activation='relu')(x)
        outputs = x
        model = keras.Model(inputs=inputs, outputs=outputs, name='VehicleAgent')
        model._make_predict_function() # force graph creation on main thread https://stackoverflow.com/questions/46725323/keras-tensorflow-exception-while-predicting-from-multiple-threads/46757715#46757715
        return model

    def mutate(self, alpha):
        for layer in range(0, self.weights.__len__(), 2): # step of 2 to skip the biases
            for row in range(0, self.weights[layer].__len__()):
                for col in range(0, self.weights[layer][row].__len__()):
                    self.weights[layer][row][col] = np.random.normal(self.weights[layer][row][col], alpha)
        return self

    def crossover(self, other, alpha):
        for layer in range(0, self.weights.__len__(), 2): # step of 2 to skip the biases
            for row in range(0, self.weights[layer].__len__()):
                for col in range(0, self.weights[layer][row].__len__()):
                    a = random.gauss(0.5, alpha)
                    self.weights[layer][row][col] = a * self.weights[layer][row][col] + (1 - a) * other.weights[layer][row][col] # blend recombination
        return self

    def clone(self):
        clone = Individual(
            max_steps=self.max_steps,
            weights=copy.deepcopy(self.weights))
        return clone

    def load(self, filename):
        import pickle
        self.weights = pickle.load(open(filename + "/weights.pickle", "rb"))

    def save(self, filename):
        import keras
        import tensorflow as tf
        from tensorflow.python.framework import graph_util
        from tensorflow.python.platform import gfile
        
        model = Individual.create_model()
        model.set_weights(self.weights)
        
        # add some variables to match tf2bc api
        tf.Variable(
            1,
            name="is_continuous_control",
            trainable=False,
            dtype=tf.int32,
        )
        tf.Variable(
            2,
            name="version_number",
            trainable=False,
            dtype=tf.int32,
        )
        tf.Variable(0, name="memory_size", trainable=False, dtype=tf.int32)
        tf.Variable(
            2,
            name="action_output_shape",
            trainable=False,
            dtype=tf.int32,
        )
        
        graph = tf.get_default_graph()
        graph_def = graph.as_graph_def()
        session = keras.backend.get_session()
        output_graph_def = graph_util.convert_variables_to_constants(
            session, graph_def, [node.op.name for node in model.outputs]   # https://stackoverflow.com/questions/40028175/
        )
        frozen_graph_def_path = filename + "/frozen_graph_def.pb"

        import os
        import errno
        if not os.path.exists(os.path.dirname(frozen_graph_def_path)):  # https://stackoverflow.com/questions/12517451/
            try:
                os.makedirs(os.path.dirname(frozen_graph_def_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with gfile.GFile(frozen_graph_def_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        
        from mlagents.trainers import tensorflow_to_barracuda as tf2bc
        tf2bc.convert(frozen_graph_def_path, frozen_graph_def_path + ".nn")

        import pickle
        pickle.dump(self.weights, open(filename + "/weights.pickle", "wb" ))

    def experience_env(self, env):
        model = Individual.create_model()
        model.set_weights(self.weights)

        self.fitness = 0
        self.rewards = None
        self.stepcount = 0
        self.starttime = timer()
        observations = env.reset()
        while True:
            actions = [model.predict(np.array([observation,]))[0] for observation in observations] # model.predict(np.array([observation,]))[0] :: expects ndarray, not array, and outputs ndarray
            observations, rewards, dones, infos = env.step(actions)
            
            rewards = np.reshape(np.array(rewards),(len(rewards),1))

            if(self.rewards is None):
                self.rewards = rewards
            else:
                self.rewards = np.hstack((self.rewards, rewards))

            self.stepcount += 1

            if self.rewards.ndim > 1:
                self.fitness = sum(sum(self.rewards))

            if self.step_callback is not None:
                self.step_callback(self)

            if(any(dones)): # agent has crashed. if the nn crashes on any corner it is no good so we can stop right away.
                self.fitness = -1
                break

            if(self.stepcount > self.max_steps):
                break

        if self.step_callback is not None:
            self.step_callback(self)

        self.endtime = timer()