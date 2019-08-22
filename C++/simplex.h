// Solves a linear programming problem with a basic simplex implementation.
//
// Implements the simplex algorithm, exactly as specified in Procedure 13.1 in
// Nocedal & Wright - Numerical Optimization (2006, 2nd ed., Springer).
//
// Attempts to solve the LP problem
//
//          min c'*x,  subject to A*x = b, x >= 0                         (13.1)
//
// starting from the initial point x0. Note that this function is for
// educational purposes only, and does not solve LP problems fast. It will not
// give reliable results for LPs that are poorly scaled, sufficiently large, or
// otherwise nontrivial.
//
// This function is written to closely match Procedure 13.1 (page 370), in which
// p is a column of the matrix B -- not the index of the variable entering the
// basis. The text above Procedure 13.1 is not clear on whether to treat p as a
// variable index or a column number in B.
//
// Note that Example 13.1 on page 371 contains a number of typos (see the
// handout), and that simplex will not give output that matches the example
// exactly.
//
// Parameters:
//   const Eigen::VectorXf &c: Cost in the objective function
//   const Eigen::MatrixXf &A: Constraint matrix
//   const Eigen::VectorXf &b: Constraint right-hand side
//   const Eigen::VectorXf &x0: Initial point
//   bool report: Whether to print a verbose output report at each iteration
//   int indexing: Set to 1 to print indices starting from 1 (to match textbook)
//   int iterlim: iteration limit (default is the number of possible bases)
//
// Returns:
//   Struct Output with members
//     Eigen::MatrixXf x: Optimal point x^* (a/the solution to the LP).
//     float fval: The optimal objective function value c' x^*.
//     Eigen::MatrixXf iterates: Sequence of points visited.
//     Eigen::MatrixXi bases: Sequence of basis indices (corresponding to
//         visited points).
//     Eigen::MatrixXf lambda: vector of multipliers for A*x = b.
//     Eigen::MatrixXf s: reduced costs of the variables in the solution x.
//     int exitflag: 1 if simplex terminates at a solution, 0 if reaching the
//         iteration limit, -3 if detecting the LP is unbounded.
//
// Throws:
//   invalid_argument: The initial point x0 is invalid -- x0 has the wrong
//       number of zero elements or x0 is infeasable.
//
//
// Tor Heirung, last modified August 2019.            https://github.com/heirung

#ifndef SIMPLEX_H
#define SIMPLEX_H

#include <Eigen/Dense>

// Struct returned by simplex()
struct Output {
    Eigen::MatrixXf x;
    float fval;
    Eigen::MatrixXf iterates;
    Eigen::MatrixXi bases;
    Eigen::VectorXf lambda;
    Eigen::VectorXf s;
    int exitflag;
};

// Solves a linear program in standard form --- main function
Output simplex(const Eigen::VectorXf &c, const Eigen::MatrixXf &A,
               const Eigen::VectorXf &b, const Eigen::VectorXf &x0,
               bool report=false, int indexing=0, int iterlim=-1);

// Reports details from every iteration (calls the other print_report())
void print_report(int k, const Eigen::ArrayXi &basic,
                  const Eigen::ArrayXi &nbasic, const Eigen::VectorXf &x,
                  const Eigen::VectorXf &xB, const Eigen::VectorXf &la,
                  const Eigen::VectorXf &sN, float fval, bool opt, int indexing,
                  int q, const Eigen::VectorXf &d, float x_q_plus, int p,
                  int var_leaving_B, const Eigen::VectorXf &xB_plus,
                  const Eigen::VectorXf &xN_plus);

// Reports details from every iteration (called by the other print_report())
void print_report(int k, const Eigen::ArrayXi &basic,
                  const Eigen::ArrayXi &nbasic, const Eigen::VectorXf &x,
                  const Eigen::VectorXf &xB, const Eigen::VectorXf &la,
                  const Eigen::VectorXf &sN, float fval, bool opt,
                  int indexing);

// Returns indices that satisfy boolean condition in array
Eigen::ArrayXi where(Eigen::Array<bool, Eigen::Dynamic, 1> cond, int nElem=-1);

// Determines the binomial coefficient "n choose k"
inline int nchoosek(int n, int k);

// Determines n factorial (n!)
inline int factorial(int n);


#endif // SIMPLEX_H
