import numpy as np

class DifferentialEvolution:
    def __init__(self, objective_function, bounds, population_size=50, generations=100, 
                 w=0.5, p=0.9, random_seed=None, **kwargs):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = population_size
        self.generations = generations
        self.w = w
        self.p = p
        if random_seed is not None: np.random.seed(random_seed)
        self.best_fitness_history = []
        self.population_history = [] 

    def optimize(self, verbose=True):
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))
        fitness = np.array([self.objective_function(ind) for ind in pop])
        
        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_sol = pop[best_idx].copy()
        
        self.population_history.append(pop.copy())

        for gen in range(self.generations):
            new_pop = np.zeros_like(pop)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                
                z = a + self.w * (b - c)
                z = np.clip(z, self.bounds[:, 0], self.bounds[:, 1])
                
                cross = np.random.rand(self.dim) < self.p
                if not np.any(cross): cross[np.random.randint(0, self.dim)] = True
                candidate = np.where(cross, z, pop[i])
                
                f_cand = self.objective_function(candidate)
                if f_cand < fitness[i]:
                    new_pop[i] = candidate
                    fitness[i] = f_cand
                    if f_cand < best_val:
                        best_val = f_cand
                        best_sol = candidate.copy()
                else:
                    new_pop[i] = pop[i]
            
            pop = new_pop
            self.best_fitness_history.append(best_val)
            self.population_history.append(pop.copy())
            
        return best_sol, best_val

    def get_history(self):
        return {'best_fitness': self.best_fitness_history, 'population': self.population_history}