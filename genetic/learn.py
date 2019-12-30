
if __name__ == '__main__':    

    from individual import Individual
    from env_manager import EnvManager
    from instrumentation import PlotManager

    maxSteps = 500
    logdir = "summariesga/run1/"

    class LearningParameters:
        pass

    params = LearningParameters()
    params.s = 2 # linear ranking selection value
    params.populationsize = 50
    params.mutationsize = 0.5
    params.generations = 50

    generation = []

    for _ in range(0,params.populationsize):
        generation.append(Individual(max_steps=maxSteps).mutate(0.5))

    envmanager = EnvManager("envs/Windows/PoD.exe", 8)
    #envmanager = EnvManager(None, 1)
    envmanager.start_workers()

    plots = PlotManager()
    #plots.begin_worker()
    #plots.instrument_individuals(generation)


    for gid in range(0,params.generations):

        print("Evaluating (" + str(gid) + ")")

        envmanager.test_generation(generation)
        try:
            envmanager.wait()
        except KeyboardInterrupt: # handle CTRL+C
            break

        # crossover

        generation.sort(key=lambda x: x.fitness, reverse=False) # ranking

        for i in range(0,len(generation)):
            generation[i].selectionprobability = (2 - params.s) / params.populationsize + (2 * i * (params.s - 1)) / (params.populationsize * (params.populationsize - 1))
        generation.sort(key=lambda x: x.selectionprobability, reverse=True)

        for individual in generation:
            print("Fitness (" + str(gid) + "): " + str(individual.fitness) + " " + "Selection: " + str(individual.selectionprobability) + " " + "Elapsed: " + str(individual.endtime-individual.starttime))

        parents = []
        for individual in generation:
            for _ in range(0,round(params.populationsize * individual.selectionprobability)):
                parents.append(individual.clone())

        generation = parents[0:params.populationsize]
        for i in range(1,len(generation)):
            generation[i].mutate(params.mutationsize) # asexual reproduction with elitism

    print("Done")

    envmanager.close()
    plots.close()