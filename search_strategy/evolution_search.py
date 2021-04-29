import time
import numpy as np

import torch
import torch.nn as nn


def evoluation_algorithm(
        trainer,
        training_strategy,
        supernet,
        val_loader,
        lookup_table,
        target_hc,
        logger,
        generation_num=20,
        population=60,
        parent_num=30,
        info_metric="flops"):
    # Population initialization
    new_population = []
    population_info = []
    for p in range(population):
        architecture = training_strategy.generate_training_architecture()
        architecture_info = lookup_table.get_model_info(
            architecture, info_metric=info_metric)

        while architecture_info > target_hc:
            architecture = training_strategy.generate_training_architecture()
            architecture_info = lookup_table.get_model_info(
                architecture, info_metric=info_metric)
        new_population.append(architecture.tolist())
        population_info.append(architecture_info)

    new_population = np.array(new_population)
    population_fitness = get_population_accuracy(
        new_population, trainer, supernet, val_loader, info_metric)
    population_info = np.array(population_info)

    # Generation start
    global_best_fitness = 0
    start_time = time.time()
    for g in range(generation_num):
        logger.info(
            "Generation : {}, Time : {}".format(
                g, time.time() - start_time))
        cur_best_fitness = np.max(population_fitness)

        if global_best_fitness < cur_best_fitness:
            global_best_fitness = cur_best_fitness
            logger.info(
                "New global best fitness : {}".format(global_best_fitness))

        parents, parents_fitness = select_mating_pool(
            new_population, population_fitness, parent_num)
        offspring_size = population - parent_num

        evoluation_id = 0
        offspring = []
        while evoluation_id < offspring_size:
            # Evolve for each offspring
            offspring_evolution = crossover(parents)
            offspring_evolution = mutation(
                offspring_evolution, training_strategy)

            offspring_hc = lookup_table.get_model_info(
                offspring_evolution, info_metric=info_metric)

            if offspring_hc <= target_hc:
                offspring.append(offspring_evolution)
                evoluation_id += 1

        offspring_evolution = np.array(offspring_evolution)
        offspring_fittness = get_population_accuracy(
            offspring_evolution, trainer, supernet, val_loader, info_metric)

        new_population[:parent_num, :] = parents
        new_population[parent_num:, :] = offspring_evolution

        population_fitness[:parent_num] = parents_fitness
        population_fitness[parent_num:] = offspring_fittness

    best_match_index = np.argmax(population_fitness)
    logger.info("Best fitness : {}".format(np.max(population_fitness)))

    return new_population[best_match_index]


def select_mating_pool(population, population_fitness, parent_num):
    pf_sort_indexs = population_fitness.argsort()[::-1]
    pf_indexs = pf_sort_indexs[:parent_num]

    parents = population[pf_indexs]
    parents_fitness = population_fitness[pf_indexs]

    return parents, parents_fitness


def crossover(parents):
    parents_size = parents.shape[0]
    architecture_len = parents.shape[1]

    offspring_evolution = np.empty((1, architecture_len), dtype=np.int32)

    crossover_point = np.random.randint(low=0, high=architecture_len)
    parent1_idx = np.random.randint(low=0, high=parents_size)
    parent2_idx = np.random.randint(low=0, high=parents_size)

    offspring_evolution[0,
                        :crossover_point] = parents[parent1_idx,
                                                    :crossover_point]
    offspring_evolution[0,
                        crossover_point:] = parents[parent2_idx,
                                                    crossover_point:]
    return offspring_evolution


def mutation(offspring_evolution, training_strategy):
    architecture_len = offspring_evolution.shape[1]

    for l in range(architecture_len):
        mutation_p = np.random.choice([0, 1], p=[0.9, 0.1])

        if mutation_p == 1:
            # Mutation activate
            micro_len = training_strategy.get_block_len()
            random_mutation = np.random.randint(low=0, high=micro_len)

            offspring_evolution[0, l] = random_mutation
    return offspring_evolution


def get_population_accuracy(
        population,
        trainer,
        supernet,
        val_loader,
        info_metric="flops"):
    architectures_top1_acc = []
    for architecture in population:
        supernet.module.set_activate_architecture(architecture) if isinstance(
            supernet, nn.DataParallel) else supernet.set_activate_architecture(architecture)
        architectures_top1_acc.append(
            trainer.validate(
                supernet, val_loader, 0))

    return np.array(architectures_top1_acc)
