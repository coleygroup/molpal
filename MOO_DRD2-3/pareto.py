import numpy as np
import scipy.stats

# physbo/search/pareto.py (https://github.com/issp-center-dev/PHYSBO)

class Rectangles(object):
    def __init__(self, n_dim, dtype):
        """
        Initilize a set of hyper-rectangle.
        :param n_dim: dimension of rectangles
        """
        self.n_dim = n_dim
        self.lb = np.zeros((0, self.n_dim), dtype=dtype)
        self.ub = np.zeros((0, self.n_dim), dtype=dtype)

    def add(self, lb, ub):
        """
        Add new rectangles.
        :param lb: lower bounds of rectangles
        :param ub: upper bounds of rectangles
        """
        self.lb = np.r_[self.lb, lb]
        self.ub = np.r_[self.ub, ub]

def dominate(t1, t2):
    """domination rule for maximization problem"""
    return np.all(t1 >= t2) and np.any(t1 > t2)

class Pareto(object):
    def __init__(self, num_objectives, dom_rule=None):
        self.num_objectives = num_objectives
        self.front = np.zeros((0, self.num_objectives))
        self.front_num = np.zeros(0, dtype=int)
        self.num_compared = 0
        self.dom_rule = dom_rule
        self.front_updated = False

        if self.dom_rule is None:
            self.dom_rule = dominate

        self.cells = Rectangles(num_objectives, int)
        self.reference_min = None
        self.reference_max = None

    def update_front(self, t):
        """
        Update the non-dominated set of points.
        Pareto set is sorted on the first objective in ascending order.
        """
        t = np.array(t)
        if t.ndim == 1:
            tt = [t]
        else:
            tt = t

        front_updated = False

        for k in range(len(tt)):
            point = tt[k]
            is_front = True
            for i in range(len(self.front)):
                if self.dom_rule(self.front[i], point):
                    is_front = False
                    break

            if is_front:
                front_updated = True
                dom_filter = np.full(len(self.front), True, dtype=bool)
                for i in range(len(self.front)):
                    if self.dom_rule(point, self.front[i]):
                        dom_filter[i] = False

                self.front = np.r_[self.front[dom_filter], point[np.newaxis, :]]
                self.front_num = np.r_[self.front_num[dom_filter], self.num_compared]

            self.num_compared += 1

        if front_updated:
            sorted_idx = self.front[:, 0].argsort()
            self.front = self.front[sorted_idx, :]
            self.front_num = self.front_num[sorted_idx]
            self.divide_non_dominated_region()

        self.front_updated = front_updated

    def export_front(self):
        return self.front, self.front_num

    def set_reference_min(self, reference_min=None):
        if reference_min is None:
            # estimate reference min point
            front_min = np.min(self.front, axis=0, keepdims=True)
            w = np.max(self.front, axis=0, keepdims=True) - front_min
            reference_min = front_min - w * 2 / self.front.shape[0]

        self.reference_min = reference_min

    def set_reference_max(self, reference_max=None):
        if reference_max is None:
            # estimate reference max point
            front_max = np.max(self.front, axis=0, keepdims=True)
            w = front_max - np.min(self.front, axis=0, keepdims=True)
            reference_max = front_max + w * 100

        self.reference_max = reference_max

    def volume_in_dominance(self, ref_min, ref_max, dominance_ratio=False):
        ref_min = np.array(ref_min)
        ref_max = np.array(ref_max)
        v_all = np.prod(ref_max - ref_min)

        front = np.r_[[ref_min], self.front, [ref_max]]
        ax = np.arange(self.num_objectives)
        lb = front[self.cells.lb, ax]
        ub = front[self.cells.ub, ax]

        v_non_dom = np.sum(np.prod(ub - lb, axis=1))

        if dominance_ratio:
            return (v_all - v_non_dom) / v_all
        else:
            return v_all - v_non_dom

    def divide_non_dominated_region(self, force_binary_search=False):
        # clear rectangles
        self.cells = Rectangles(self.num_objectives, int)

        if self.num_objectives == 2 and not force_binary_search:
            self.__divide_2d()
        else:
            self.__divide_using_binary_search()

    def __divide_2d(self):
        """
        Divide non-dominated region into vertical rectangles for the case of 2-objectives.
        Assumes that Pareto set has been sorted on the first objective in ascending order.
        Notes:
            In 2-dimensional cases, the second objective has be sorted in decending order.
        """
        n_cells = self.front.shape[0] + 1
        lb_idx = [[i, (i + 1) % n_cells] for i in range(n_cells)]
        ub_idx = [[i + 1, n_cells] for i in range(n_cells)]

        self.cells.add(lb_idx, ub_idx)

    def __included_in_non_dom_region(self, p):
        return np.all([np.any(pf <= p) for pf in self.front])

    def __divide_using_binary_search(self):
        front = np.r_[
            np.full((1, self.num_objectives), -np.inf),
            self.front,
            np.full((1, self.num_objectives), np.inf),
        ]

        # Pareto front indices when sorted on each dimension's front value in ascending order.
        # (indices start from 1)
        # Index 0 means anti-ideal value, index `self.front.shape[0] + 1` means ideal point.
        front_idx = np.r_[
            np.zeros((1, self.num_objectives), dtype=int),
            np.argsort(self.front, axis=0) + 1,
            np.full((1, self.num_objectives), self.front.shape[0] + 1, dtype=int),
        ]

        rect_candidates = [[np.copy(front_idx[0]), np.copy(front_idx[-1])]]

        while rect_candidates:
            rect = rect_candidates.pop()

            lb_idx = [front_idx[rect[0][d], d] for d in range(self.num_objectives)]
            ub_idx = [front_idx[rect[1][d], d] for d in range(self.num_objectives)]
            lb = [front[lb_idx[d], d] for d in range(self.num_objectives)]
            ub = [front[ub_idx[d], d] for d in range(self.num_objectives)]

            if self.__included_in_non_dom_region(lb):
                self.cells.add([lb_idx], [ub_idx])

            elif self.__included_in_non_dom_region(ub):
                rect_sizes = rect[1] - rect[0]

                # divide rectangle by the dimension with largest size
                if np.any(rect_sizes > 1):
                    div_dim = np.argmax(rect_sizes)
                    div_point = rect[0][div_dim] + int(round(rect_sizes[div_dim] / 2.0))

                    # add divided left rectangle
                    left_ub_idx = np.copy(rect[1])
                    left_ub_idx[div_dim] = div_point
                    rect_candidates.append([np.copy(rect[0]), left_ub_idx])

                    # add divided right rectangle
                    right_lb_idx = np.copy(rect[0])
                    right_lb_idx[div_dim] = div_point
                    rect_candidates.append([right_lb_idx, np.copy(rect[1])])


