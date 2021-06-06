import time
import numpy as np

import torch
import torch.nn as nn

from .base import BaseSearcher

class EvolutionSearcher(BaseSearcher):
    def __init__(self, config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger):
        super(EvolutionSearcher, self).__init__(config, supernet, val_loader, lookup_table, training_strategy, device, criterion, logger)

        self.generation_num = self.config["search_utility"]["generation_num"]
        self.population_num = self.config["search_utility"]["population_num"]
        self.parent_num = self.config["search_utility"]["parent_num"]

    def step(self):
        pass

    def search(self):
        # Population initialization
        new_population = []
        population_info = []

        for p in range(self.population_num):
            architecture = self.training_strategy.generate_training_architecture()
            architecture_info = self.lookup_table.get_model_info(architecture)

            while architecture_info > self.target_hc:
                architecture = self.training_strategy.generate_training_architecture()
                architecture_info = self.lookup_table.get_model_info(architecture)

            new_population.append(architecture.tolist())
            population_info.append(architecture_info)

        new_population = np.array(new_population)
        population_fitness = self.evaluate_architectures(new_population)
        population_info = np.array(population_info)

        # Generation start
        global_best_fitness = 0
        start_time = time.time()
        for g in range(self.generation_num):
            self.logger.info(
                "Generation : {}, Time : {}".format(
                    g, time.time() - start_time))
            cur_best_fitness = np.max(population_fitness)

            if global_best_fitness < cur_best_fitness:
                global_best_fitness = cur_best_fitness
                self.logger.info(
                    "New global best fitness : {}".format(global_best_fitness))

            parents, parents_fitness = self.select_mating_pool(
                new_population, population_fitness, self.parent_num)
            offspring_size = self.population_num - self.parent_num

            evoluation_id = 0
            offspring_evolution = []
            while evoluation_id < offspring_size:
                # Evolve for each offspring
                offspring = self.crossover(parents)
                offspring = self.mutation(offspring)

                offspring_hc = self.lookup_table.get_model_info(offspring[0])

                if offspring_hc <= self.target_hc:
                    offspring_evolution.extend(offspring)
                    evoluation_id += 1

            offspring_evolution = np.array(offspring_evolution)
            offspring_fittness = self.evaluate_architectures(offspring_evolution)

            new_population[:self.parent_num, :] = parents
            new_population[self.parent_num:, :] = offspring_evolution

            population_fitness[:self.parent_num] = parents_fitness
            population_fitness[self.parent_num:] = offspring_fittness

        best_match_index = np.argmax(population_fitness)
        self.logger.info("Best fitness : {}".format(np.max(population_fitness)))

        best_architecture = new_population[best_match_index]
        best_architecture_top1 = population_fitness[best_match_index]
        best_architecture_hc = self.lookup_table.get_model_info(best_architecture)

        return best_architecture, best_architecture_hc, best_architecture_top1


    def select_mating_pool(self, population, population_fitness, parent_num):
        pf_sort_indexs = population_fitness.argsort()[::-1]
        pf_indexs = pf_sort_indexs[:parent_num]

        parents = population[pf_indexs]
        parents_fitness = population_fitness[pf_indexs]

        return parents, parents_fitness


    def crossover(self, parents):
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


    def mutation(self, offspring_evolution):
        architecture_len = offspring_evolution.shape[1]

        for l in range(architecture_len):
            mutation_p = np.random.choice([0, 1], p=[0.9, 0.1])

            if mutation_p == 1:
                # Mutation activate
                micro_len = self.training_strategy.get_block_len()
                random_mutation = np.random.randint(low=0, high=micro_len)

                offspring_evolution[0, l] = random_mutation
        return offspring_evolution

