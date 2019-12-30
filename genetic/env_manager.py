import threading
import gym
from gym_unity.envs import UnityEnv
from multiprocessing import Process, Manager


# Runs a set of persistent environments in separate threads
# Based on the queue example
# https://docs.python.org/3/library/queue.html#module-queue
class EnvManager:   
    
    @staticmethod    
    def worker(environment_filename, workerq, q, results):
        worker_id = workerq.get()
        env = UnityEnv(environment_filename = environment_filename, multiagent = True, worker_id=worker_id)
        workerq.task_done()

        while True:
            individual = q.get()
            if individual is None:
                break

            individual.experience_env(env)
            results.put(individual) # re-pickle with updated member variables

            q.task_done()

        env.close()

    def start_workers(self):
        self.workerq = self.manager.Queue()
        for x in range(0, self.num_workers):
            self.workerq.put(x)

        self.processes = []
        for _ in range(0, self.num_workers):
            p = Process(target=self.worker, args=(self.environment_filename, self.workerq, self.q, self.results))
            p.start()
            self.processes.append(p)

        self.workerq.join() # wait for all unity environments to connect before proceeding

    def test_generation(self, generation):
        self.items.clear()
        for g in generation:
            self.test_individual(g)

    def test_individual(self, individual):
        individual.guid = self.counter
        self.counter += 1
        self.items[individual.guid] = individual
        self.q.put(individual)

    def wait(self):
        self.q.join()
        self.results.join()

    def results_worker(self):
        while True:
            result = self.results.get()
            if result is None:
                break
            self.items[result.guid].__dict__.update(result.__dict__)
            self.results.task_done()

    def close(self):
        for _ in range(self.num_workers):
            self.q.put(None)
        for p in self.processes:
            p.join()
        self.results.put(None)
        self.resultsthread.join()

    def __init__(self, environment_filename, num_workers):
        self.manager = Manager()
        self.q = self.manager.Queue()
        self.results = self.manager.Queue()
        self.items = {}
        self.num_workers = num_workers
        self.environment_filename = environment_filename
        self.counter = 0
        self.resultsthread = threading.Thread(target=self.results_worker)
        self.resultsthread.start()