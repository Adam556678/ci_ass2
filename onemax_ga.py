from genetic_algorithm import GeneticAlgorithm
import random
import time
import math
import numpy as np
from typing import List

class OneMaxGA(GeneticAlgorithm):
    """
    Genetic algorithm for solving the One-Max problem.
    Inherits from the GeneticAlgorithm abstract base class.
    """

    def __init__(self, population_size: int, chromosome_length: int, crossover_prob:float, mutation_rate: float, elitism_num: int, min:int, max:int):
        """
        Initialize the OneMaxGA instance.

        Args:
            population_size (int): Size of the population.
            chromosome_length (int): Length of each chromosome (bitstring).
            mutation_rate (float): Probability of mutation for each bit.
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.elitism_num = elitism_num
        self.population = self.initialize_population()
        self.min = min
        self.max = max

    def create_individual(self) -> List[int]:
        """
        Create a new individual (random bitstring).

        Returns:
            List[int]: A newly created individual.
        """
        return [random.randint(0, 1) for _ in range(self.chromosome_length)]    
    
    def initialize_population(self) -> List[List[int]]:
        """
        Initialize the population with random bitstrings.

        Returns:
            List[List[int]]: Initial population.
        """

        return [self.create_individual() for _ in range(self.population_size)]
    
    def standard_decoder(self, chromosome: List[int]):
        chromosome_length = len(chromosome)
        
        sum = 0
        for i in range(chromosome_length):
            sum += chromosome[i] * (2**(chromosome_length-i-1))
        
        chromosome_real_val = self.min + (sum / 2 ** chromosome_length) * (self.max - self.min)
        return math.ceil(chromosome_real_val)

    def evaluate_fitness(self, chromosome: List[int]) -> int:
        chromosome_x1_value = self.standard_decoder(chromosome=chromosome[:2])
        chromosome_x2_value = self.standard_decoder(chromosome=chromosome[2:])
        
        fitness = 8 - ((chromosome_x1_value + 0.0317) ** 2) + (chromosome_x2_value ** 2)
        return fitness

    def linear_rank_selection(self, pop: List[List[int]], sp: float):
        population_size = len(pop)
        population_fitness = [self.evaluate_fitness(indiv) for indiv in pop]
        
        ranks = np.array(population_fitness).argsort().argsort() + 1
        pop_linear_rank_fitness = [(2-sp) + 2 * (sp - 1) * (rank-1)/(population_size-1) for rank in ranks]
        return pop_linear_rank_fitness
        
    
    def calculate_cumulative_probabilities(self, pop_fitness) -> List[float]:
        """
        Calculate cumulative probabilities for each individual.

        Returns:
            List[float]: Cumulative probabilities.
        """
        total_fitness = sum(fit for fit in pop_fitness)
        probabilities = [fit / total_fitness for fit in pop_fitness]
        cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(pop_fitness))]
        return cumulative_probabilities

    def select_parents(self,) -> List[List[int]]:
        """
        Select parents based on cumulative probabilities.

        Returns:
            List[List[int]]: Selected parents.
        """
        sp = random.uniform(1,2)
        pop_linear_rank_fitness = self.linear_rank_selection(self.population,sp)
        cumulative_probabilities = self.calculate_cumulative_probabilities(pop_fitness=pop_linear_rank_fitness)
        selected_parents = random.choices(self.population, cum_weights = cumulative_probabilities, k = 2)
        return selected_parents

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[List[int]]:
        """
        Perform one-point crossover between two parents.

        Args:
            parent1 (List[int]): First parent chromosome.
            parent2 (List[int]): Second parent chromosome.

        Returns:
            List[List[int]]: Two offspring chromosomes.
        """
        

        if random.uniform(0, 1) < self.crossover_prob:
            rand = random.uniform(0,self.chromosome_length)
            crossover_site = int(rand)
            
            temp = parent1.copy()
            parent1 = parent1[:crossover_site+1] + parent2[crossover_site+1 : self.chromosome_length]
            parent2 = parent2[:crossover_site+1] + temp[crossover_site+1 : self.chromosome_length]

        return parent1, parent2
    

    def mutate(self, chromosome: List[int]) -> List[int]:
        """
        Apply bit flip mutation to an individual.

        Args:
            chromosome (List[int]): The chromosome to be mutated.

        Returns:
            List[int]: The mutated chromosome.
        """
        mutated_chromosome = chromosome.copy()
        for i in range(self.chromosome_length):
            if random.uniform(0, 1) < self.mutation_rate:
                if mutated_chromosome[i] == 1:
                    mutated_chromosome[i] = 0
                else:
                    mutated_chromosome[i] = 1
        return mutated_chromosome

    def elitism(self) -> List[List[int]]:
        """
        Apply elitism to the population (keep the best two individuals).

        Args:
            new_population (List[List[int]]): The new population after crossover and mutation.
        """
        sorted_population = sorted(self.population, key=self.evaluate_fitness, reverse=True)
        best_individuals = sorted_population[:self.elitism_num]
        return best_individuals


    def run(self, max_generations):
        for generation in range(max_generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents()
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                new_population.extend([offspring1, offspring2])

            new_population = new_population[0:self.population_size-self.elitism_num] # make sure the new_population is the same size of original population - the best individuals we will append next
            best_individuals = self.elitism()
            new_population.extend(best_individuals)
            self.population = new_population


        best_solution = max(self.population, key=self.evaluate_fitness)
        return best_solution

if __name__ == "__main__":
    population_size = 100
    chromosome_length = 4
    crossover_prob = 0.7
    mutation_rate = 0.07
    elitism_num = 2
    max_generations = 150
    variable_min = -2
    variable_max = 2
    start = time.time()
    onemax_ga = OneMaxGA(population_size, chromosome_length,crossover_prob, mutation_rate,elitism_num, min=variable_min, max=variable_max)
    best_solution = onemax_ga.run(max_generations)
    ga_time = time.time()-start
    print("GA Solution Time:",round(ga_time,1),'Seconds')


    print(f"Best solution: {best_solution}")
    print(f"Fitness: {onemax_ga.evaluate_fitness(best_solution)}")
