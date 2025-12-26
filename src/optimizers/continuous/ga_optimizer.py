import numpy as np
from typing import Callable, List, Tuple

class GeneticAlgorithm:
    def __init__(
        self,
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_size: int = 2,
        selection_method: str = 'tournament',  
        tournament_size: int = 3,
        truncation_size: int = 10,            
        random_seed: int = None
    ):
        """
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) tuples
            population_size: Number of individuals
            generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elite_size: Number of best individuals to preserve
            selection_method: 'tournament', 'roulette', or 'truncation'
            tournament_size: Size of tournament (for tournament selection)
            truncation_size: Number of top individuals (for truncation selection)
            random_seed: Random seed
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        self.selection_method = selection_method.lower()
        self.tournament_size = tournament_size
        self.truncation_size = truncation_size
        
        if random_seed is not None:
            np.random.seed(random_seed)

        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.population_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self):
        """Create initial random population within bounds."""
        population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.population_size, self.dimensions)
        )
        return population
    
    def evaluate_population(self, population):
        fitness = np.array([self.objective_function(ind) for ind in population])
        return fitness

    def tournament_selection(self, population, fitness):
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness[indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()

    def truncation_selection(self, population, fitness):
        sorted_indices = np.argsort(fitness)
        top_k_indices = sorted_indices[:self.truncation_size]
        selected_idx = np.random.choice(top_k_indices)
        return population[selected_idx].copy()

    def roulette_wheel_selection(self, population, fitness):
        max_f = np.max(fitness)
        weights = max_f - fitness + 1e-6

        total_weight = np.sum(weights)
        if total_weight == 0:
            probs = np.ones(len(fitness)) / len(fitness)
        else:
            probs = weights / total_weight
        selected_idx = np.random.choice(len(population), p=probs)
        return population[selected_idx].copy()

    def select_parent(self, population, fitness):
        if self.selection_method == 'roulette':
            return self.roulette_wheel_selection(population, fitness)
        elif self.selection_method == 'truncation':
            return self.truncation_selection(population, fitness)
        else:
            return self.tournament_selection(population, fitness)


    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            mask = np.random.random(self.dimensions) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        for i in range(self.dimensions):
            if np.random.random() < self.mutation_rate:
                mutation_range = self.bounds[i, 1] - self.bounds[i, 0]
                mutation = np.random.normal(0, 0.1 * mutation_range)
                individual[i] += mutation
                individual[i] = np.clip(individual[i], self.bounds[i, 0], self.bounds[i, 1])
        return individual
    
    def optimize(self, verbose=True):
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        for generation in range(self.generations):
            best_idx = np.argmin(fitness)
            self.best_fitness_history.append(fitness[best_idx])
            self.mean_fitness_history.append(np.mean(fitness))
            self.population_history.append(population.copy())
            
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = population[best_idx].copy()
            
            if verbose and (generation % 10 == 0 or generation == self.generations - 1):
                print(f"Generation {generation}: Best Fitness = {fitness[best_idx]:.6f}")

            new_population = []

            if self.elite_size > 0:
                elite_indices = np.argsort(fitness)[:self.elite_size]
                for idx in elite_indices:
                    new_population.append(population[idx].copy())

            while len(new_population) < self.population_size:
                parent1 = self.select_parent(population, fitness)
                parent2 = self.select_parent(population, fitness)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = np.array(new_population[:self.population_size])
            fitness = self.evaluate_population(population)
        
        return self.best_solution, self.best_fitness
    
    def get_history(self):
        return {
            'best_fitness': self.best_fitness_history,
            'mean_fitness': self.mean_fitness_history,
            'population': self.population_history
        }