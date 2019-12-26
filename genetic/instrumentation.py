import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class PlotManager:
    def __init__(self):
        self.threads = []
        self.closing = False
        self.frames = queue.Queue()
        #self.writer = SummaryWriter() # https://tensorboardx.readthedocs.io/en/latest/tutorial.html#what-is-tensorboard-x

    def close(self):
        self.closing = True
        for t in self.threads:
            t.join()

    def instrument_individual(self, individual):
        individual.step_callback = lambda x: self.step_callback(x)

    def instrument_individuals(self, individuals):
        for individual in individuals:
            self.instrument_individual(individual)

    class Frame(object):
        pass

    # can be called from different threads
    def step_callback(self, individual):
        frame = self.Frame()
        frame.individual = individual
        frame.rewards = individual.rewards[:,-1]
        frame.fitness = individual.fitness
        frame.step = individual.stepcount
        self.frames.put(frame) 

    def begin_worker(self):
        def worker(): # https://stackoverflow.com/questions/18791722/can-you-plot-live-data-in-matplotlib
            plt.figure()
            plt.ion()
            plt.show()
            series = {}
            lateststepcount = 0
            maxy = 0

            while True:
                if self.closing:
                    break

                frame = self.frames.get()

                individual = frame.individual
                lateststepcount = max(lateststepcount,frame.step)

                if individual not in series:
                    series[frame.individual] = plt.scatter([],[])
                    
                dataseries = series[individual]

                x = np.repeat(frame.step, frame.rewards.shape[0])
                y = frame.rewards
                a = np.vstack((x,y))
                b = np.transpose(a)

                

                existing = dataseries.get_offsets() # https://stackoverflow.com/questions/40686697/python-matplotlib-update-scatter
                if existing.size > 0:
                    dataseries.set_offsets(np.vstack((existing, b)))
                else:
                    dataseries.set_offsets(b)

                maxy = max((maxy, y.max()))
                plt.ylim(-1,maxy)
                plt.xlim(0,lateststepcount)

                if self.frames.empty():
                    plt.draw()
                    plt.pause(0.5) # for the gui
               
        thread = threading.Thread(target=worker)
        thread.start()
        self.threads.append(thread)
