#!/usr/bin/python3

"""
Solves a linear program with more constraints than the one in Example 13.1 in 
Nocedal & Wright - Numerical Optimization (2006, 2nd ed., Springer), using an 
implementation of the simplex algorithm as specified in Procedure 13.1. Prints 
a detatiled explanation from every iteration. Solves a Phase-I problem to 
determine a basic feasible starting point x0. 

Tor Heirung, last modified March 2019.               https://github.com/heirung
"""


import numpy as np

from simplex import (simplex, phase_I)


# Specify the LP and convert to standard form: 
A_ineq = np.array([ 
    [0, 1],   # <=  9
    [1, 2],   # <= 20
    [1, 3],   # <= 28
    [1, 1],   # <= 13
    [2, 1],   # <= 20
    [3, 1],   # <= 28
    [1, 0] ]) # <=  9 
m = A_ineq.shape[0] # m inequality constraints 
A = np.hstack((A_ineq, np.eye(m)))
b = np.array([9, 20, 28, 13, 20, 28, 9])
c = np.hstack((np.array([-1.0, -1.1]), np.zeros((m))))
n = len(c) # n variables in standard form

# Solve a Phase-I problem to determine a basic feasible starting point x0
# (could in this simple case also use x0 = np.hstack((np.zeros(n-m), b))):
x0, _ = phase_I(A, b)

# Solve with simplex (Procedure 13.1), starting from x0: 
indexing = 1; # print with 1 as base index, like in the texbook
x, output = simplex(c, A, b, x0, report=True, indexing=indexing)

print('Points visited in the x1-x2 plane:') 
print(output['iterates'][:2, :]) 
print(f'Sequence of basic variables ({indexing}-indexed):') 
print(output['bases'] + indexing) 