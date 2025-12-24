import numpy as np
from .ga_optimizer import GeneticAlgorithm

class HybridGAPSO(GeneticAlgorithm):
    def __init__(self, objective_function, bounds, population_size=50, generations=100, 
                 crossover_rate=0.8, mutation_rate=0.1, w=0.5, c1=1.5, c2=1.5, **kwargs):
        super().__init__(objective_function, bounds, population_size, generations, 
                         crossover_rate, mutation_rate, **kwargs)
        self.w = w; self.c1 = c1; self.c2 = c2
        self.population_history = [] # <--- [ADD]

    def optimize(self, verbose=True):
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.population_size, self.dimensions))
        fit = self.evaluate_population(pop)
        
        vel = np.zeros_like(pop)
        p_best_pos = pop.copy(); p_best_val = fit.copy()
        g_best_idx = np.argmin(fit)
        g_best_pos = pop[g_best_idx].copy(); g_best_val = fit[g_best_idx]
        
        self.best_fitness_history = []
        self.population_history.append(pop.copy()) # [ADD] Lưu ban đầu

        for gen in range(self.generations):
            # GA Part
            next_pop = []
            elite_idx = np.argsort(fit)[:self.elite_size]
            for idx in elite_idx: next_pop.append(pop[idx].copy())
            
            while len(next_pop) < self.population_size:
                p1 = self.tournament_selection(pop, fit)
                p2 = self.tournament_selection(pop, fit)
                c1, c2 = self.crossover(p1, p2)
                next_pop.extend([c1, c2])
            
            pop = np.array(next_pop[:self.population_size])
            
            # PSO Part
            r1 = np.random.rand(self.population_size, self.dimensions)
            r2 = np.random.rand(self.population_size, self.dimensions)
            vel = self.w * vel + self.c1 * r1 * (p_best_pos - pop) + self.c2 * r2 * (g_best_pos - pop)
            pop = pop + vel
            pop = np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])
            
            fit = self.evaluate_population(pop)
            
            mask = fit < p_best_val
            p_best_pos[mask] = pop[mask]; p_best_val[mask] = fit[mask]
            
            min_idx = np.argmin(p_best_val)
            if p_best_val[min_idx] < g_best_val:
                g_best_val = p_best_val[min_idx]
                g_best_pos = p_best_pos[min_idx].copy()
                
            self.best_fitness_history.append(g_best_val)
            self.population_history.append(pop.copy()) # [ADD] Lưu mỗi thế hệ
            
        return g_best_pos, g_best_val
    
    def get_history(self):
        return {'best_fitness': self.best_fitness_history, 'population': self.population_history}