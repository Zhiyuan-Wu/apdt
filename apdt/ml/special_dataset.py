# pylint: disable=E1101
import numpy as np

class sin_regression():
    '''Sin regression meta dataset. Generate random functions of form f(x)=Asin(wx+phi)

    Parameters
    -----
        s_size: int, default 5, support sample number.
        q_size: int, default 20, query sample number.
        dataset_size: tuple of int, default (10000, 2000, 10000), number of functions in train/val/test set.
        A_range: tuple of int, default (0.1, 5.0), uniform random range of parameter A.
        phi_range: tuple of int, default (0.0, np.pi), uniform random range of parameter phi.
        w_range: tuple of int, default (8.0, 12.0), uniform random range of parameter w.
        *_range_test: parameter range of test set, default to be same as train/val. in case of you need train-test shift.
    '''
    def __init__(self, **kwarg):
        if "s_size" not in kwarg.keys():
            kwarg["s_size"] = 5
        if "q_size" not in kwarg.keys():
            kwarg["q_size"] = 20
        if "dataset_size" not in kwarg.keys():
            kwarg["dataset_size"] = (10000, 2000, 10000)
        if "A_range" not in kwarg.keys():
            kwarg["A_range"] = (0.1, 5.0)
        if "phi_range" not in kwarg.keys():
            kwarg["phi_range"] = (0.0, np.pi)
        if "w_range" not in kwarg.keys():
            kwarg["w_range"] = (8.0, 12.0)
        if "A_range_test" not in kwarg.keys():
            kwarg["A_range_test"] = kwarg["A_range"]
        if "phi_range_test" not in kwarg.keys():
            kwarg["phi_range_test"] = kwarg["phi_range"]
        if "w_range_test" not in kwarg.keys():
            kwarg["w_range_test"] = kwarg["w_range"]
        self.kwarg = kwarg
        self.tr_batch_num = kwarg["dataset_size"][0]
        self.val_batch_num = kwarg["dataset_size"][1]
        self.te_batch_num = kwarg["dataset_size"][2]
    
    def get_data(self, A, phi, w, batch_size):
        x_s = np.random.rand(batch_size, self.kwarg["s_size"], 1)
        y_s = A * np.sin(x_s * w + phi)
        x_q = np.random.rand(batch_size, self.kwarg["q_size"], 1)
        y_q = A * np.sin(x_q * w + phi)
        return [x_s, y_s, x_q, y_q]
    
    def tr_get_batch(self, batch_size=1):
        A = np.random.rand(batch_size, 1, 1) * (self.kwarg["A_range"][1] - self.kwarg["A_range"][0]) + self.kwarg["A_range"][0]
        phi = np.random.rand(batch_size, 1, 1) * (self.kwarg["phi_range"][1] - self.kwarg["phi_range"][0]) + self.kwarg["phi_range"][0]
        w = np.random.rand(batch_size, 1, 1) * (self.kwarg["w_range"][1] - self.kwarg["w_range"][0]) + self.kwarg["w_range"][0]
        return self.get_data(A, phi, w, batch_size)

    def val_get_batch(self, batch_size=1):
        A = np.random.rand(batch_size, 1, 1) * (self.kwarg["A_range"][1] - self.kwarg["A_range"][0]) + self.kwarg["A_range"][0]
        phi = np.random.rand(batch_size, 1, 1) * (self.kwarg["phi_range"][1] - self.kwarg["phi_range"][0]) + self.kwarg["phi_range"][0]
        w = np.random.rand(batch_size, 1, 1) * (self.kwarg["w_range"][1] - self.kwarg["w_range"][0]) + self.kwarg["w_range"][0]
        return self.get_data(A, phi, w, batch_size)

    def te_get_batch(self, batch_size=1):
        A = np.random.rand(batch_size, 1, 1) * (self.kwarg["A_range_test"][1] - self.kwarg["A_range_test"][0]) + self.kwarg["A_range_test"][0]
        phi = np.random.rand(batch_size, 1, 1) * (self.kwarg["phi_range_test"][1] - self.kwarg["phi_range_test"][0]) + self.kwarg["phi_range_test"][0]
        w = np.random.rand(batch_size, 1, 1) * (self.kwarg["w_range_test"][1] - self.kwarg["w_range_test"][0]) + self.kwarg["w_range_test"][0]
        return self.get_data(A, phi, w, batch_size)
