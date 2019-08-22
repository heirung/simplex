# simplex
Bare-bones Python 3 and C++ implementations of the simplex algorithm, written to closely mirror Procedure 13.1 in Nocedal & Wright - Numerical Optimization (2006, 2nd ed., Springer). 

In Python, run simplex_example_13_1.py to print a step-by-step report with details on every iteration. 
Note that Example 13.1 on page 371 contains a number of typos (see the handout - Example_13_1.pdf) and that the output from this simplex implementation will not match the textbook example exactly. 
Run simplex_example_7_constraints.py for a slightly larger example problem (seven constraints), which solves a Phase-I problem to determine a basic feasible starting point. 

In C++, compile simplex.cpp with either simplex_example_13_1.cpp or simplex_example_7_constraints.cpp using a standard compiler and run the resulting executable. 
The C++ version uses [Eigen](http://eigen.tuxfamily.org) for linear algebra. 
