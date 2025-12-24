import numpy as np
import math

class CuckooSearch:
    def __init__(self, objective_function=None, bounds=None, 
                 n_nests=50, max_iter=100, pa=0.25, beta=1.5, alpha=0.01, 
                 random_seed=None, **kwargs):
        """
        Cuckoo Search Optimizer (Standardized for Lab 03)
        """
        self.objective_function = objective_function
        
        # Xử lý bounds: đảm bảo là numpy array
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
        self.pa = pa          # Discovery rate of alien eggs/solutions
        self.beta = beta      # Levy flight parameter
        self.alpha = alpha    # Step size scaling factor
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.best_fitness_history = []
        self.population_history = [] # <--- QUAN TRỌNG: Để vẽ hình 2D

    def _levy_flight(self, u):
        """Hàm sinh bước nhảy Levy"""
        sigma_u = (math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2) / 
                   (math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        sigma_v = 1
        
        u_vec = np.random.normal(0, sigma_u, size=self.dim)
        v_vec = np.random.normal(0, sigma_v, size=self.dim)
        
        step = u_vec / (np.abs(v_vec) ** (1 / self.beta))
        return step

    def optimize(self, objective_func=None, dim=None, bounds=None, verbose=True):
        # Hỗ trợ cả cách gọi cũ (truyền tham số vào optimize) và mới (đã init)
        if self.objective_function is None and objective_func is not None:
            self.objective_function = objective_func
        if self.bounds is None and bounds is not None:
            self.bounds = np.array(bounds)
            self.dim = len(bounds)
            self.lb = self.bounds[:, 0]
            self.ub = self.bounds[:, 1]

        # 1. Khởi tạo quần thể (Nests)
        nests = np.random.uniform(self.lb, self.ub, (self.n_nests, self.dim))
        fitness = np.array([self.objective_function(x) for x in nests])
        
        # Tìm nest tốt nhất ban đầu
        best_idx = np.argmin(fitness)
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Lưu trạng thái ban đầu
        self.population_history = []
        self.best_fitness_history = []
        self.population_history.append(nests.copy())

        for it in range(self.max_iter):
            # 2. Tạo nest mới bằng Levy Flight
            new_nests = nests.copy()
            
            for i in range(self.n_nests):
                # Chọn random nest khác để làm base
                step_size = self.alpha * self._levy_flight(self.beta) * (nests[i] - best_nest)
                new_nests[i] += step_size * np.random.randn(self.dim)
                
                # Clip bounds
                new_nests[i] = np.clip(new_nests[i], self.lb, self.ub)

            # Đánh giá và Selection (Greedy)
            new_fitness = np.array([self.objective_function(x) for x in new_nests])
            improved = new_fitness < fitness
            nests[improved] = new_nests[improved]
            fitness[improved] = new_fitness[improved]
            
            # 3. Loại bỏ các nest xấu (Abandonment) với xác suất pa
            # Thay thế bằng random nests mới (hoặc local search tùy biến thể, đây dùng random walk)
            worst_nests_indices = np.random.rand(self.n_nests) < self.pa
            if np.any(worst_nests_indices):
                # Tạo bước nhảy random từ 2 nest bất kỳ
                step_size = np.random.rand() * (nests[np.random.randint(0, self.n_nests, self.n_nests)] - 
                                              nests[np.random.randint(0, self.n_nests, self.n_nests)])
                
                new_nests = nests.copy()
                new_nests[worst_nests_indices] += step_size[worst_nests_indices]
                
                # Clip bounds
                new_nests = np.clip(new_nests, self.lb, self.ub)
                
                # Đánh giá lại
                new_fitness = np.array([self.objective_function(x) for x in new_nests])
                
                # Greedy selection lần 2
                improved = new_fitness < fitness
                nests[improved] = new_nests[improved]
                fitness[improved] = new_fitness[improved]

            # 4. Cập nhật Global Best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_nest = nests[min_idx].copy()
            
            # 5. Lưu history
            self.best_fitness_history.append(best_fitness)
            self.population_history.append(nests.copy()) # <--- QUAN TRỌNG
            
        return best_nest, best_fitness

    def get_history(self):
        return {
            'best_fitness': self.best_fitness_history,
            'population': self.population_history # <--- Trả về để vẽ
        }