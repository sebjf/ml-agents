import numpy as np
import tensorflow as tf
import queue
import threading
import gym
from gym_unity.envs import UnityEnv

import matplotlib.pyplot as plt

maxSteps = 500


class Individual:
    def create_model(self):
        inputs = tf.keras.Input(shape=(43,)) #=env.observation_space.shape
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(2, activation='relu')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='VehicleAgent')
        model._make_predict_function() # force graph creation on main thread https://stackoverflow.com/questions/46725323/keras-tensorflow-exception-while-predicting-from-multiple-threads/46757715#46757715
        self.model = model

    def __init__(self):
        self.create_model()
    
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

    def experience_env(self, env):
        self.fitness = 0
        self.rewards = None
        self.stepcount = 0
        observations = env.reset()
        while True:
            actions = self.evaluate_model(observations)
            observations, rewards, dones, infos = env.step(actions)
            
            if(self.rewards is None):
                self.rewards = np.array(rewards)
            else:
                self.rewards = np.vstack((self.rewards, np.array(rewards)))

            self.stepcount += 1

            if(self.stepcount > maxSteps):
                break
            
            if(any(dones)): # agent has crashed. if the nn crashes on any corner it is no good so we can stop right away.
                self.fitness = -1
                break

        self.rewards += 1 #bias slightly in case the car, e.g. rolled backwards a little
        self.fitness = sum(sum(self.rewards))



# Runs a set of persistent environments in separate threads
# Based on the queue example
# https://docs.python.org/3/library/queue.html#module-queue
class EnvManager:       
    def worker(self):
        worker_id = self.workerq.get()
        env = UnityEnv(environment_filename = None, multiagent = True, worker_id=worker_id)
        self.workerq.task_done()

        while True:
            individual = self.q.get()
            if individual is None:
                break
            individual.experience_env(env)
            self.q.task_done()

        env.close()

    def start_workers(self):
        self.workerq = queue.Queue()
        for x in range(0,self.num_workers):
            self.workerq.put(x)

        self.threads = []
        for i in range(0, self.num_workers):
            t = threading.Thread(target=self.worker)
            t.start()
            self.threads.append(t)

        self.workerq.join() # wait for all unity environments to connect before proceeding

    def test_generation(self, generation):
        for g in generation:
            self.test_individual(g)

    def test_individual(self, individual):
        self.q.put(individual)

    def wait(self):
        self.q.join()

    def close(self):
        for i in range(self.num_workers):
            self.q.put(None)
        for t in self.threads:
            t.join()

    def __init__(self, num_workers):
        self.q = queue.Queue()
        self.num_workers = num_workers

class PlotManager:
    def __init__(self):
        self.threads = []
        self.closing = False

    def close(self):
        self.closing = True
        for t in self.threads:
            t.join()
        
    def begin_plot_rewards(self,individuals):
        def worker(): # https://stackoverflow.com/questions/18791722/can-you-plot-live-data-in-matplotlib
            plt.figure()
            plt.ion()
            plt.show()
            series = {}
            while True:
                if self.closing:
                    break

                xlim = 0
                ymin = 0
                ymax = 1

                for individual in individuals:
                    if "rewards" in dir(individual) and individual.rewards is not None: # race condition
                        if individual not in series:
                            series[individual] = plt.scatter([],[])

                        # reformat to match the expectations of set_offsts
                        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html
                        # https://stackoverflow.com/questions/34280444/python-scatter-plot-with-multiple-y-values-for-each-x
                        # https://stackoverflow.com/questions/40686697/python-matplotlib-update-scatter

                        x = np.tile(range(0,individual.rewards.shape[0]), individual.rewards.shape[1])
                        y = np.reshape(individual.rewards, len(x), order='F')
                        xlim = np.max((xlim, x.max()))
                        ymin = np.min((ymin, y.min()))
                        ymax = np.max((ymax, y.max()))
                        d = np.transpose(np.vstack((x,y)))
                        series[individual].set_offsets(d)

                plt.xlim(0,xlim)
                plt.ylim(-1,ymax)
                plt.draw()
                plt.pause(0.5)
               
        thread = threading.Thread(target=worker)
        thread.start()
        self.threads.append(thread)

generation = []
generation.append(Individual().mutate(0.5))
generation.append(Individual().mutate(0.5))
generation.append(Individual().mutate(0.5))
generation.append(Individual().mutate(0.5))

plots = PlotManager()
plots.begin_plot_rewards(generation)

envmanager = EnvManager(1)
envmanager.start_workers()

envmanager.test_generation(generation)
envmanager.wait()

plots.close()
envmanager.close()