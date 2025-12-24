# simulation.py
from ga import (
    initialize_population,
    evaluate_fitness,
    tournament_selection,
    crossover,
    mutate
)
import random
import time

def run_ga_generator(tasks, vms, servers,
                     pop_size=30,
                     generations=50,
                     mutation_rate=0.05,
                     crossover_points=1,
                     elitism=1,
                     time_limit=None):
    """
    Generator that yields after each generation:
    yields: (gen_index, best_chromosome, best_fitness, fitness_history_list)
    """
    num_tasks = len(tasks)
    num_vms = len(vms)
    population = initialize_population(pop_size, num_tasks, num_vms)
    start = time.time()
    history = []
    tempgen = 0
    for gen in range(generations):

        # fitness evaluation
        fitnesses = [evaluate_fitness(ch, tasks, vms, servers) for ch in population]
        # store best
        best_idx = int(min(range(len(fitnesses)), key=lambda i: fitnesses[i]))
        best_chrom = population[best_idx][:]
        best_fit = fitnesses[best_idx]
        history.append(best_fit)
        tempgen += 1
        # yield current result so GUI can update
        yield gen, best_chrom, best_fit, history[:]
        # تحقق من الوقت بعد كل جيل
        if time_limit is not None and (time.time() - start) >= time_limit:
            print(f"Time limit reached at generation {gen + 1}")
            break
        # elitism: keep top `elitism` individuals
        sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        elites = [population[i][:] for i in sorted_idx[:elitism]]

        # selection
        selected = tournament_selection(population, fitnesses, k=3)

        # create next generation
        next_pop = elites[:]
        while len(next_pop) < pop_size:
            p1, p2 = random.choice(selected), random.choice(selected)
            c1, c2 = crossover(p1, p2, points=crossover_points)
            c1 = mutate(c1, num_vms, mutation_rate)
            c2 = mutate(c2, num_vms, mutation_rate)
            next_pop.append(c1)
            if len(next_pop) < pop_size:
                next_pop.append(c2)

        population = next_pop

        # if time_limit and (time.time() - start) >= time_limit:
        #     break

    # final evaluation and yield final
    fitnesses = [evaluate_fitness(ch, tasks, vms, servers) for ch in population]
    best_idx = int(min(range(len(fitnesses)), key=lambda i: fitnesses[i]))
    best_chrom = population[best_idx][:]
    best_fit = fitnesses[best_idx]
    history.append(best_fit)
    yield tempgen, best_chrom, best_fit, history[:]




