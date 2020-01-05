from individual import Individual
from env_manager import EnvManager
from instrumentation import PlotManager
import random

maxSteps = 500

def control():
    envmanager = EnvManager(None, 1)
    envmanager.start_workers()
    individual = Individual(max_steps=10000)
    individual.load('models/ga/pb')
    while True:
        envmanager.test_individual(individual)
        envmanager.wait()

def main():
    class LearningParameters:
        pass

    params = LearningParameters()
    params.populationsize = 5  
    params.mutationsize = 0.2   # std deviation of normal distribution used to mutate a weight
    params.mutationrate = 0.1
    params.generations = 50     # how long to run for
    params.crossoversize = 0.5  # std deviation of normal distribution determining the weighting between two weights during crossover

    generation = []

    for _ in range(0,params.populationsize):
        generation.append(Individual(max_steps=maxSteps).mutate(params.mutationsize)) # initial mutation is larger

    #envmanager = EnvManager("envs/Windows/PoD.exe", 8)
    envmanager = EnvManager(None, 1)
    envmanager.start_workers()

    plots = PlotManager(tensorboarddir='summaries/ga1')
    #plots.begin_worker()
    #plots.instrument_individuals(generation)

    bestIndividual = None

    for gid in range(0,params.generations):

        print("Evaluating (" + str(gid) + ")")

        envmanager.test_generation(generation)
        try:
            envmanager.wait()
        except KeyboardInterrupt: # handle CTRL+C
            break

        # crossover

        generation.sort(key=lambda x: x.fitness, reverse=False) # ranking

        K = params.populationsize

        for i in range(0,len(generation)):
            if i <= K / 2:
                generation[i].selectionprobability = (12 * i) / (5 * K * (K + 2))
            else:
                generation[i].selectionprobability = (28 * i) / (5 * K * ((3 * K) + 2))

        generation.sort(key=lambda x: x.selectionprobability, reverse=True)

        for individual in generation:
            print("Fitness (" + str(gid) + "): " + str(individual.fitness) + " " + "Selection: " + str(individual.selectionprobability) + " " + "Elapsed: " + str(individual.endtime-individual.starttime))

        plots.plot_generation_fitness(generation, gid)
        plots.plot_generation_rewards(generation, gid)
        plots.flush()

        N = 100
        parents = []
        for individual in generation:
            for _ in range(0,round(N * K * individual.selectionprobability)):
                parents.append(individual)

        # create the new popualtion. binary reproduction with elitism

        bestIndividual = generation[0].clone()

        generation = []
        generation.append(bestIndividual)

        for i in range(0,K-1):
            if random.random() > params.mutationrate:
                individual = random.choice(parents).clone().crossover(random.choice(parents), 0.5)
            else:
                individual = random.choice(parents).mutate(params.mutationsize)
            generation.append(individual)

    print("Done")

    if bestIndividual is not None:
        bestIndividual.save("models/ga1/pb")

    envmanager.close()
    plots.close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        main()