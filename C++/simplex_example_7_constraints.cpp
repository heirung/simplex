// Solves a linear program with more constraints than the one in Example 13.1 in
// Nocedal & Wright - Numerical Optimization (2006, 2nd ed., Springer), using an
// implementation of the simplex algorithm as specified in Procedure 13.1.
// Prints a detatiled explanation from every iteration.
//
// Tor Heirung, last modified August 2019.            https://github.com/heirung

#include "simplex.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>
using namespace std;
using namespace Eigen;


int main() {
    
    // Specify the LP and convert to standard form: 
    const int m = 7; // number of equality constraints
    const int n_ineq = 2, n = n_ineq + m; // number of variables
    MatrixXf A_ineq(m, n_ineq), A(m, n);
    VectorXf c_ineq(n_ineq), c(n), b(m), x0(n);
    c_ineq << -1, -1.1;
    A_ineq << 0, 1,
              1, 2,
              1, 3,
              1, 1,
              2, 1,
              3, 1,
              1, 0;
    c.head(n_ineq) = c_ineq;
    c.tail(m).setZero();
    A.leftCols(n_ineq) = A_ineq;
    A.rightCols(m).setIdentity();
    b << 9, 20, 28, 13, 20, 28, 9;
    
    x0.head(n - m).setZero(); // basic feasible starting point
    x0.tail(m) = b; // x0 = [0, ... 0, b']'

    bool report = false;
    int indexing = 0;
    Output result = simplex(c, A, b, x0, report, indexing);

    cout << "Points visited in the x1-x2 plane: \n"
         << result.iterates.topRows<2>() << endl;

    cout << "Sequence of basic variables (" << indexing << "-indexed): \n" 
         << result.bases.array() + indexing << "\n\n";
    
    return 0;
}
