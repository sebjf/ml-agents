import numpy as np
import tensorflow as tf
import gym

class Individual:
    def create_model(self):
        inputs = tf.keras.Input(shape=(44,)) #=env.observation_space.shape
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(2, activation='relu')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='VehicleAgent')
        model._make_predict_function() # force graph creation on main thread https://stackoverflow.com/questions/46725323/keras-tensorflow-exception-while-predicting-from-multiple-threads/46757715#46757715
        self.model = model

    def __init__(self):
        self.create_model()
        self.step_callback = None
    
    def evaluate_model(self, observations):
        return [self.model.predict(np.array([observation,]))[0] for observation in observations] # model.predict(np.array([observation,]))[0] :: expects ndarray, not array, and outputs ndarray

    def mutate(self, alpha):
        weights = self.model.get_weights()
        for layer in range(0, weights.__len__(), 2): # step of 2 to skip the biases
            for row in range(0, weights[layer].__len__()):
                for col in range(0, weights[layer][row].__len__()):
                    weights[layer][row][col] = np.random.normal(weights[layer][row][col], alpha)
        self.model.set_weights(weights)
        return self

    def clone(self):
        clone = Individual()
        clone.model.set_weights(self.model.get_weights())
        return clone

    def experience_env(self, env, max_steps):
        self.fitness = 0
        self.rewards = None
        self.stepcount = 0
        observations = env.reset()
        while True:
            actions = self.evaluate_model(observations)
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

            if(self.stepcount > max_steps):
                break

        if self.step_callback is not None:
            self.step_callback(self)