import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, objective_function, bounds, population_size=50, generations=100, 
                 w=0.7, c1=1.5, c2=1.5, random_seed=None, **kwargs):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = population_size
        self.max_iter = generations
        self.w = w; self.c1 = c1; self.c2 = c2
        if random_seed is not None: np.random.seed(random_seed)
        
        self.best_fitness_history = []
        self.population_history = []

    def optimize(self, verbose=True):
        X = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))
        V = np.zeros_like(X)
        P_best_pos = X.copy()
        P_best_val = np.array([self.objective_function(x) for x in X])
        
        g_best_idx = np.argmin(P_best_val)
        g_best_pos = P_best_pos[g_best_idx].copy()
        g_best_val = P_best_val[g_best_idx]

        self.population_history.append(X.copy()) 
        
        v_max = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])

        for i in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            V = self.w * V + self.c1 * r1 * (P_best_pos - X) + self.c2 * r2 * (g_best_pos - X)
            V = np.clip(V, -v_max, v_max)
            X = X + V
            X = np.clip(X, self.bounds[:, 0], self.bounds[:, 1])
            
            current_fit = np.array([self.objective_function(x) for x in X])
            
            better_mask = current_fit < P_best_val
            P_best_pos[better_mask] = X[better_mask]
            P_best_val[better_mask] = current_fit[better_mask]
            
            min_idx = np.argmin(P_best_val)
            if P_best_val[min_idx] < g_best_val:
                g_best_val = P_best_val[min_idx]
                g_best_pos = P_best_pos[min_idx].copy()
            
            self.best_fitness_history.append(g_best_val)
            self.population_history.append(X.copy())
            
        return g_best_pos, g_best_val

    def get_history(self):
        return {'best_fitness': self.best_fitness_history, 'population': self.population_history}