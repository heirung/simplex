// simplex.cpp
// Tor Heirung, last modified August 2019.            https://github.com/heirung

#include "simplex.h"
#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/LU>
using namespace std;
using namespace Eigen;


Output simplex(const VectorXf &c, const MatrixXf &A, const VectorXf &b,
               const VectorXf &x0, bool report, int indexing, int iterlim) {
    
    // Number of variables n and constraints m (standard form: Ax = b, x >= 0)
    const int n = (int) A.cols();
    const int m = (int) A.rows();
    
    // Default iteration limit: max number of bases
    iterlim = (iterlim == -1) ? nchoosek(n, m) : iterlim;
    
    if (!b.isApprox(A * x0)) {
        throw invalid_argument("x0 is infeasible: A * x0 != b");
    } else if ((x0.array() < 0).any()) {
        throw invalid_argument("x0 is infeasible: negative components.");
    } else if ((x0.array() == 0).count() != n - m) {
        throw invalid_argument("Invalid x0: x0 must contain n-m zero elements.");
    } // checking the basis below
    
    // Basic and nonbasic index sets: (complements of each other)
    ArrayXi nbasic(n - m), basic(m);
    // Can always assume m < n, otherwise A rank deficient (fix: p. 358)
    int nb_cnt(0), b_cnt(0);
    for (int i = 0; i < n; i++) {
        if (x0(i) == 0 && nb_cnt < n - m) {
            nbasic(nb_cnt++) = i;
        } else {
            basic(b_cnt++) = i;
        }
    }
    
    // Partition A, x, and c, and determine lambda (la) and sN
    MatrixXf B(m, m), N(m, n - m);
    VectorXf xB(m), xB_plus(m), cB(m), xN(n - m), xN_plus(n - m), cN(n - m);
    for (int i = 0; i < m; i++) {
        B.col(i) = A.col(basic[i]);
        xB(i) = x0(basic[i]);
        cB(i) = c(basic[i]);
    }
    for (int i = 0; i < n - m; i++) {
        N.col(i) = A.col(nbasic[i]);
        xN(i) = x0(nbasic[i]);
        cN(i) = c(nbasic[i]);
    }
    
    FullPivLU<MatrixXf> lu_decomp(B); // to determine rank(B)
    if (lu_decomp.rank() < m) { // cf. eq. (13.13)
        throw invalid_argument("The initial basis matrix B is singular!");
    }
    if ((xB.array() == 0).any()) { // cf. Definition 13.1
        throw invalid_argument("The initial basis is degenerate.");
    }
    
    VectorXf x(x0), la(m), sN(n - m), d(m), ratios;
    ArrayXi i_d_g_0;
    MatrixXf::Index i_min_sN, i_p, iN_q;
    int q, p, var_leaving_B, exitflag;
    float fval, x_q_plus, min_sN;
    exitflag = numeric_limits<int>::infinity();
    fval = numeric_limits<float>::infinity();
    
    MatrixXf iterates = MatrixXf::Constant(n, iterlim, -1.0);
    MatrixXi bases = MatrixXi::Constant(m, iterlim, -1);
    iterates.col(0) = x;
    bases.col(0) = basic;
    
    bool opt(false);
    int k(0);
    while (!opt) {
        
        // lambda: multiplier for Ax = b, B'*la = cB
        la = (B.transpose()).fullPivLu().solve(cB);
        // s: multiplier for x >= 0 (sN: "reduced cost" of xN)
        sN = cN - N.transpose() * la;
        
        fval = c.transpose() * x;
        
        if ((sN.array() >= 0).all()) { // optimal point found
            opt = true;
            exitflag = 1;
            if (report)
                print_report(k, basic, nbasic, x, xB, la, sN, fval, opt, indexing);
            break; // trim iterates and bases and return
        }
        
        // Pivot: first determine the variable x_q that will enter the basis
        min_sN = sN.minCoeff(&i_min_sN); // find a neg. sN component (s_q < 0)
        q = nbasic[i_min_sN]; // x_q will enter the basis (q is a variable ind.)
        d = B.fullPivLu().solve(A.col(q)); // solve B*d = A_q for d
        if ((d.array() < 0).all()) {
            exitflag = -3;
            cout << "Problem unbounded. Stopped.\n";
            break; // trim iterates and bases and return
        }
        
        i_d_g_0 = where(d.array() > 0); // indices of the positive elements in d
        ratios.resize(i_d_g_0.size());
        for (int i = 0; i < i_d_g_0.size(); i++) {
            ratios(i) = xB(i_d_g_0(i)) / d(i_d_g_0(i));
        }
        x_q_plus = ratios.minCoeff(&i_p); // minimum over i | d_i > 0 (Dantzig)
        // p is the minimizing i -- will be removed from basis (NOT x_p!):
        p = i_d_g_0(i_p);
        
        // p is a column of B (the matrix). Find corresponding variable index:
        var_leaving_B = basic(p);
        // The variable entering the basis is x_q. Where in N (the set) is x_q?
        iN_q = where(nbasic == q)(0); // the first int has index 0
        
        // Update xB+ and xN+:
        xB_plus = xB - d * x_q_plus;
        xN_plus.setZero();
        xN_plus(iN_q) = x_q_plus;
        if (report) {
            print_report(k, basic, nbasic, x, xB, la, sN, fval, opt, indexing,
                         q, d, x_q_plus, p, var_leaving_B, xB_plus, xN_plus);
        }
        
        k++;
        
        // Update x with new values:
        for (int i = 0; i < m; i++) {
            x(basic(i)) = xB_plus(i);
        }
        for (int i = 0; i < n - m; i++) {
            x(nbasic(i)) = xN_plus(i);
        }
        
        // Update basic and nonbasic index sets.
        // Remove the basic variable corresponding to column p of B (matrix):
        nbasic(iN_q) = basic(p);
        // Add q to the basis:
        basic(p) = q;
        
        // Update the partitions of A (into N and B), c (into cN and cB), and
        // x (into xN (all zeros) and xB):
        N.col(iN_q) = B.col(p);
        B.col(p) = A.col(q);
        cN(iN_q) = cB(p);
        cB(p) = c(q);
        for (int i = 0; i < m; i++) {
            xB(i) = x(basic(i));
        }
        
        iterates.col(k) = x;
        bases.col(k) = basic;
        
        if (k >= iterlim - 1) {
            exitflag = 0;
            cout << "Iteration limit reached. Stopped\n";
            break;
        }
        
    }
    
    // Prepare output: multiplier s and Output struct
    VectorXf s = VectorXf::Zero(n);
    for (int i = 0; i < n - m; i++) {
        s(nbasic(i)) = sN(i);
    }
    Output output = {x, fval, iterates.leftCols(k + 1), bases.leftCols(k + 1),
        la, s, exitflag};
    
    return output;
}


void print_report(int k, const ArrayXi &basic, const ArrayXi &nbasic,
                  const VectorXf &x, const VectorXf &xB, const VectorXf &la,
                  const VectorXf &sN, float fval, bool opt, int indexing, int q,
                  const VectorXf &d, float x_q_plus, int p, int var_leaving_B,
                  const VectorXf &xB_plus, const VectorXf &xN_plus) {
    
    string hline = string(80, '-');
    IOFormat vec(3, 0, "", ",","", "", "[", "]'"); // also defined below
    
    if (k == 0) {
        cout << "Eigen version: " << EIGEN_WORLD_VERSION << "." <<
        EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "\n\n";
    }
        
    print_report(k, basic, nbasic, x, xB, la, sN, fval, opt, indexing);
    
    if (!opt) {
        cout << "  x =      " << x.format(vec) << endl;
        cout << "  c'x =     " << fval << endl;
        cout << "  x_" << q + indexing << " will enter the basis (q = "
             << q + indexing << ")\n";
        cout << "  d =      " << d.format(vec) << endl;
        cout << "  x_q+ = x_" << q + indexing << "+ = " << x_q_plus
             << "\t(value of entering variable/step length)\n";
        cout << "  x_" << var_leaving_B + indexing
             << " will leave the basis (position p = " << p + indexing
             << " in the basis)\n";
        cout << "  x_B+ =   " << xB_plus.format(vec)
             << "\t(current basic vector at new point)\n";
        cout << "  x_N+ =   " << xN_plus.format(vec)
             << "\t(current nonbasic vector at new point)\n";
    }
    cout << hline << "\n\n";
}


void print_report(int k, const ArrayXi &basic, const ArrayXi &nbasic,
                  const VectorXf &x, const VectorXf &xB, const VectorXf &la,
                  const VectorXf &sN, float fval, bool opt, int indexing) {
    
    string hline = string(80, '-');
    IOFormat set(0, 0, "", ",", "", "", "{", "}");
    IOFormat vec(3, 0, "", ",", "", "", "[", "]'");
    
    cout << hline << endl;
    
    cout << "  Iteration number: " << k + indexing << " (" << indexing <<
    " indexing)\n";
    cout << "  Basic index set:     " << (basic + indexing).format(set)
         << endl;
    cout << "  Non-basic index set: " << (nbasic + indexing).format(set)
         << endl;
    cout << "  x_B =    " << xB.format(vec) << endl;
    cout << "  lambda = " << la.format(vec) << endl;
    cout << "  s_N =    " << sN.format(vec) << endl;
    
    if (opt) {
        cout << "\n  OPTIMAL POINT FOUND\n";
        cout << "  x^* =    " << x.format(vec) << endl;
        cout << "  c'x^* =   " << fval << endl;
        cout << hline << "\n\n";
    }
}


ArrayXi where(Array<bool, Dynamic, 1> cond, int nElem) {
    // Not checking for nElem < 0, nElem > cond.size(), and more.
    int nTrue = (int) cond.count(); // number of TRUEs
    // If nElem is not passed (default value: -1), default is "all indices".
    if (nElem == -1)
        nElem = nTrue;
    VectorXi indices;
    if (nTrue == 0)
        return indices; // empty
    int i_max = min(nTrue, nElem); // in case nTrue < nElem
    indices.resize(i_max);
    int i(0), i_count(0);
    while (i_count < i_max) { // not reached if nTrue == 0
        if (cond(i)) {
            indices(i_count++) = i;
        }
        i++;
    }
    return indices;
}


inline int nchoosek(int n, int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}


inline int factorial(int n) {
    // 1! = 1, 0! = 1.
    // Not handling n < 0 explicitly (but returning 1 for n < 0 avoids infinite
    // recursion.
    return (n <= 1) ? 1 : n * factorial(n - 1);
}
