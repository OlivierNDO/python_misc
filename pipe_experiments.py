# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:23:17 2020

@author: user
"""


from functools import partial, reduce
import numpy as np



# http://tomerfiliba.com/blog/Infix-Operators/

class Infix(object):
    def __init__(self, func):
        self.func = func
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return Infix(partial(self.func, other))
    def __call__(self, v1, v2):
        return self.func(v1, v2)
    

@Infix
def add(x, y):
    return x + y

@Infix
def subset(x, y):
    return [z for z in x if z in y]






temp = [10, 20, 30, 40, 50]
    
temp |subset| [20, 30] |add| 10

10 |add| 10
    
    
    
    
    

class Pipe:
    def __init__(self, obj):
        self.obj = obj
    
    def apply(self, *functions):
        return reduce(lambda x, y: y(x), functions, self.obj)



def multiply_by_six(obj):
    return obj * 6
    


Pipe([1,2,3]).apply(np.product).apply(multiply_by_six)


Pipe()
