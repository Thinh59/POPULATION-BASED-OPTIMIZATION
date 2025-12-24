import numpy as np

class WheelersRidge:
    def __init__(self, dimensions=2):
        """
        Wheeler's Ridge Function (Appendix B.7 Algorithms for Optimization)
        Một thung lũng rất cong và hẹp, cực trị tại (1, 1.5)
        """
        self.name = "Wheeler's Ridge"
        self.dimensions = dimensions
        # Giới hạn vùng tìm kiếm giống trong sách (để vẽ hình đẹp)
        self.bounds = np.array([[0.0, 3.0], [0.0, 3.0]]) 
        self.optimal_value = 0.0

    def evaluate(self, x):
        # Hàm chỉ hoạt động tốt nhất trên 2D
        # f(x) = (x1*x2 - a)^2 + (x2 - a)^2 với a = 1.5
        a = 1.5
        x1 = x[0]
        x2 = x[1]
        
        term1 = (x1 * x2 - a)**2
        term2 = (x2 - a)**2
        
        return term1 + term2

    def get_bounds(self):
        return self.bounds

    def get_name(self):
        return self.name