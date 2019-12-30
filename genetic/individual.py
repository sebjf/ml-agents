import numpy as np
import gym
from timeit import default_timer as timer

class Individual:
    def __init__(self, max_steps = 1000):
        self.step_callback = None
        self.max_steps = max_steps
        self.create_weights()

    def create_weights(self):
        model = Individual.create_model()
        self.weights = model.get_weights()

    @staticmethod
    def create_model():
        import keras
        inputs = keras.Input(shape=(44,)) #=env.observation_space.shape
        x = keras.layers.Dense(64, activation='relu')(inputs)
        outputs = keras.layers.Dense(2, activation='relu')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='VehicleAgent')
        model._make_predict_function() # force graph creation on main thread https://stackoverflow.com/questions/46725323/keras-tensorflow-exception-while-predicting-from-multiple-threads/46757715#46757715
        return model

    def mutate(self, alpha):
        for layer in range(0, self.weights.__len__(), 2): # step of 2 to skip the biases
            for row in range(0, self.weights[layer].__len__()):
                for col in range(0, self.weights[layer][row].__len__()):
                    self.weights[layer][row][col] = np.random.normal(self.weights[layer][row][col], alpha)
        return self

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