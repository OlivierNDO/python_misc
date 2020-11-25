from collections.abc import Iterable
from functools import partial, reduce
import numpy as np

class Pipe:
    def __init__(self, obj):
        self.obj = obj
    
    def apply_across(self, *function):
        return Pipe([reduce(lambda x, y: y(x), function, z) for z in self.obj])
    
    def apply(self, *function):
        return Pipe(reduce(lambda x, y: y(x), function, self.obj))
    
    def where_gt(self, x):
        return Pipe([z for z in self.obj if z > x])
    
    def where_lt(self, x):
        return Pipe([z for z in self.obj if z < x])
    
    def where_ge(self, x):
        return Pipe([z for z in self.obj if z >= x])
    
    def where_le(self, x):
        return Pipe([z for z in self.obj if z <= x])
    
    def where_in(self, x):
        return Pipe([z for z in self.obj if z in x])
    
    def where_like(self, x):
        search_str = x.replace('%', '')
        if x[-1] == '%' and x[0] == '%':
            pipe_output = Pipe([z for z in self.obj if search_str in z])
        elif x[-1] == '%':
            pipe_output = Pipe([z for z in self.obj if search_str in z[:len(search_str)]])
        elif x[0] == '%':
            pipe_output = Pipe([z for z in self.obj if search_str in z[len(search_str):]])
        else:
            pipe_output = Pipe([z for z in self.obj if search_str in z])
        return pipe_output
    
    def where(self, operator, x):
        operator_dict = {'>' : self.where_gt,
                         '<' : self.where_lt,
                         '>=' : self.where_ge,
                         '<=' : self.where_le,
                         'IN' : self.where_in,
                         'LIKE' : self.where_like}
        return operator_dict[operator.upper()](x)
    
    #def where(operator, other):
    #    return Pipe([x for x in self.obj])
    
    def get(self):
        return self.obj
    


int_list = [1,2,3,4,5]

str_list = ['apple', 'application', 'applying', 'requiring']

Pipe(str_list).where_like('ing').get()

Pipe(str_list).where('LIKE', 'app%').get()

Pipe(int_list).where('IN', [1,5]).get()

Pipe(int_list).apply(np.product).get()
