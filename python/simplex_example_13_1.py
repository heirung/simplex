#!/usr/bin/python3

"""
Solves Example 13.1 in Nocedal & Wright - Numerical Optimization (2006, 2nd 
ed., Springer), using an implementation of the simplex algorithm as specified 
in Procedure 13.1. Prints a detatiled explanation from every iteration -- see 
the handout for a list of typos in Example 13.1 in the textbook. 

Tor Heirung, last modified March 2019.               https://github.com/heirung
"""


import numpy as np

from simplex import simplex


# Specify the LP: 
c = np.array([-4, -2, 0, 0])
A = np.array([[1, 1, 1, 0],
              [2, 1/2, 0, 1]])
b = np.array([5, 8])

x0 = np.array([0, 0, 5, 8]) # basic feasible starting point

# Solve with simplex (Procedure 13.1): 
indexing = 1; # print with 1 as base index, like in the texbook
x, output = simplex(c, A, b, x0, report=True, indexing=indexing)

print('Points visited in the x1-x2 plane:') 
print(output['iterates'][:2, :]) 
print(f'Sequence of basic variables ({indexing}-indexed):') 
print(output['bases'] + indexing) 
