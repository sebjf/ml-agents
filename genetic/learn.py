
from individual import Individual
from env_manager import EnvManager
from instrumentation import PlotManager

maxSteps = 500
counter = 0
logdir = "summariesga/run1/"

generation = []
generation.append(Individual().mutate(0.5))
generation.append(Individual().mutate(0.5))
generation.append(Individual().mutate(0.5))
generation.append(Individual().mutate(0.5))

plots = PlotManager()
plots.begin_worker()

for g in generation:
    plots.instrument_individual(g)

envmanager = EnvManager("envs/Windows/PoD.exe", 4, maxSteps)
envmanager.start_workers()

envmanager.test_generation(generation)
envmanager.wait()

envmanager.close()
plots.close()