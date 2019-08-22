// Solves Example 13.1 in Nocedal & Wright - Numerical Optimization (2006, 2nd
// ed., Springer), using an implementation of the simplex algorithm as specified
// in Procedure 13.1. Prints a detatiled explanation from every iteration -- see
// the handout for a list of typos in Example 13.1 in the textbook.
//
// Tor Heirung, last modified August 2019.            https://github.com/heirung

#include "simplex.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>
using namespace std;
using namespace Eigen;


int main() {
    
    // Specify the LP:
    const int m = 2; // number of equality constraints
    const int n = 4; // number of variables
    MatrixXf A(m, n);
    VectorXf c(n), b(m), x0(n);
    c << -4, -2, 0, 0;
    A << 1, 1, 1, 0,
         2, 1.0/2, 0, 1;
    b << 5, 8;

    x0 << 0, 0, 5, 8; // basic feasible starting point

    bool report = true;
    int indexing = 1;
    Output result = simplex(c, A, b, x0, report, indexing);

    cout << "Points visited in the x1-x2 plane: \n"
         << result.iterates.topRows<2>() << endl;

    cout << "Sequence of basic variables (" << indexing << "-indexed): \n"
         << result.bases.array() + indexing << "\n\n";
    
    return 0;
}
