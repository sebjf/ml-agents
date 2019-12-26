import queue
import threading
import gym
from gym_unity.envs import UnityEnv

# Runs a set of persistent environments in separate threads
# Based on the queue example
# https://docs.python.org/3/library/queue.html#module-queue
class EnvManager:       
    def worker(self):
        worker_id = self.workerq.get()
        env = UnityEnv(environment_filename = self.environment_filename, multiagent = True, worker_id=worker_id)
        self.workerq.task_done()

        while True:
            individual = self.q.get()
            if individual is None:
                break
            individual.experience_env(env, self.max_steps)
            self.q.task_done()

        env.close()

    def start_workers(self):
        self.workerq = queue.Queue()
        for x in range(0,self.num_workers):
            self.workerq.put(x)

        self.threads = []
        for _ in range(0, self.num_workers):
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
        for _ in range(self.num_workers):
            self.q.put(None)
        for t in self.threads:
            t.join()

    def __init__(self, environment_filename, num_workers, max_steps):
        self.q = queue.Queue()
        self.num_workers = num_workers
        self.max_steps = max_steps
        self.environment_filename = environment_filename