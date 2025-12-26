import numpy as np
import math

class CuckooSearch:
    def __init__(self, objective_function=None, bounds=None, 
                 n_nests=50, max_iter=100, pa=0.25, beta=1.5, alpha=0.01, 
                 random_seed=None, **kwargs):
        self.objective_function = objective_function

        if bounds is not None:
            self.bounds = np.array(bounds)
            self.dim = len(bounds)
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]
        else:
            self.bounds = None
            self.dim = 0
            
        self.n_nests = n_nests
        self.max_iter = max_iter
        self.pa = pa          
        self.beta = beta    
        self.alpha = alpha  
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.best_fitness_history = []
        self.population_history = [] 

    def _levy_flight(self, u):
        sigma_u = (math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2) / 
                   (math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        sigma_v = 1
        
        u_vec = np.random.normal(0, sigma_u, size=self.dim)
        v_vec = np.random.normal(0, sigma_v, size=self.dim)
        
        step = u_vec / (np.abs(v_vec) ** (1 / self.beta))
        return step

    def optimize(self, objective_func=None, dim=None, bounds=None, verbose=True):
        if self.objective_function is None and objective_func is not None:
            self.objective_function = objective_func
        if self.bounds is None and bounds is not None:
            self.bounds = np.array(bounds)
            self.dim = len(bounds)
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]

        nests = np.random.uniform(self.lb, self.ub, (self.n_nests, self.dim))
        fitness = np.array([self.objective_function(x) for x in nests])

        best_idx = np.argmin(fitness)
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]

        self.population_history = []
        self.best_fitness_history = []
        self.population_history.append(nests.copy())

        for it in range(self.max_iter):
            new_nests = nests.copy()
            
            for i in range(self.n_nests):
                step_size = self.alpha * self._levy_flight(self.beta) * (nests[i] - best_nest)
                new_nests[i] += step_size * np.random.randn(self.dim)

                new_nests[i] = np.clip(new_nests[i], self.lb, self.ub)

            new_fitness = np.array([self.objective_function(x) for x in new_nests])
            improved = new_fitness < fitness
            nests[improved] = new_nests[improved]
            fitness[improved] = new_fitness[improved]
            
            worst_nests_indices = np.random.rand(self.n_nests) < self.pa
            if np.any(worst_nests_indices):
                step_size = np.random.rand() * (nests[np.random.randint(0, self.n_nests, self.n_nests)] - 
                                              nests[np.random.randint(0, self.n_nests, self.n_nests)])
                
                new_nests = nests.copy()
                new_nests[worst_nests_indices] += step_size[worst_nests_indices]

                new_nests = np.clip(new_nests, self.lb, self.ub)

                new_fitness = np.array([self.objective_function(x) for x in new_nests])

                improved = new_fitness < fitness
                nests[improved] = new_nests[improved]
                fitness[improved] = new_fitness[improved]

            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_nest = nests[min_idx].copy()
            
            self.best_fitness_history.append(best_fitness)
            self.population_history.append(nests.copy()) 
            
        return best_nest, best_fitness

    def get_history(self):
        return {
            'best_fitness': self.best_fitness_history,
            'population': self.population_history 
        }