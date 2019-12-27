
from individual import Individual
from env_manager import EnvManager
from instrumentation import PlotManager
import tensorflow as tf

maxSteps = 500
counter = 0
logdir = "summariesga/run1/"

generation = []

for _ in range(0,4):
    generation.append(Individual().mutate(0.5))

envmanager = EnvManager("envs/Windows/PoD.exe", 4, maxSteps)
envmanager.start_workers()

plots = PlotManager()
#plots.begin_worker()
#plots.instrument_individuals(generation)

for gid in range(0,5):
    envmanager.test_generation(generation)
    envmanager.wait()

    generation.sort(key=lambda x: x.fitness, reverse=True)

    for g in generation:
        print("Fitness (" + str(gid) + "): " + str(g.fitness) + " " + "Elapsed: " + str(g.endtime-g.starttime))

    for i in range(1,len(generation)):
        generation[i].mutate(0.25)

envmanager.close()
plots.close()