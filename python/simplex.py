
import numpy as np

from scipy.special import comb # "combinations, binomial coeff, N choose k"
from warnings import warn


def simplex(c, A, b, x0, report=False, indexing=0, iterlim=None):
    """Solves a linear programming problem with a basic simplex implementation.
    
    Implements the simplex algorithm, exactly as specified in Procedure 13.1 in
    Nocedal & Wright - Numerical Optimization (2006, 2nd ed., Springer).
    
    Attempts to solve the LP problem
 
             min c'*x,  subject to A*x = b, x >= 0                       (13.1)
 
    starting from the initial point x0. Note that this function is for 
    educational purposes only, and does not solve LP problems fast. It will not 
    give reliable results for LPs that are poorly scaled, sufficiently large, 
    or otherwise nontrivial. 
    
    This function is written to closely match Procedure 13.1 (page 370), in 
    which p is a column of the matrix B -- not the index of the variable 
    entering the basis. The text above Procedure 13.1 is not clear on whether 
    to treat p as a variable index or a column number in B. 
    
    Note that Example 13.1 on page 371 contains a number of typos (see the 
    handout), and that simplex will not give output that matches the example 
    exactly. 
    
    Args:
        c: Cost in the objective function
        A: Constraint matrix
        b: Constraint right-hand side
        x0: Initial point
        report: Whether to print a verbose output report at each iteration
        indexing: Set to 1 to print indices starting from 1 (to match textbook)
        iterlim: iteration limit (default is the number of possible bases)
        
    Returns:
        x: Optimal point x^* (a/the solution to the LP).
        output: Dictionary with the following keys. 
          - fval: The optimal objective function value c' x^*. 
          - iterates: Sequence of points visited.
          - bases: Sequence of basis indices (corresponding to visited points).
          - lambda: vector of multipliers for A*x = b.
          - s: reduced costs of the variables in the solution x.
          - exitflag: 1 if simplex terminates at a solution, 0 if reaching the 
            iteration limit, -3 if detecting the LP is unbounded. 
            
    Raises:
        ValueError: The initial point x0 is invalid -- x0 has the wrong number
            of zero elements or x0 is infeasable. 
            
   
    Tor Heirung, last modified March 2019.           https://github.com/heirung
    """
    
    
    n = len(c) # Number of variables (assuming standard form: Ax = b, x >= 0)
    m = len(b) # Number of constraints 
    
    if iterlim is None: 
        iterlim = comb(n, m).astype(int) # maximum number of bases
    
    iterates = np.full((n, iterlim), np.nan, dtype=float) 
    bases = np.full((m, iterlim), np.nan, dtype=int) 
        
    # Basic and nonbasic index sets: (complements of each other)
    nbasic = np.where(x0==0)[0][:n-m] # Nonbasic index set (calligraphic N)
    basic = np.setdiff1d(np.arange(n), nbasic) # Basis (calligraphic B)
    
    if not all(np.isclose(A @ x0, b)):
        raise ValueError('x0 is infeasible: A@x0 != b')
    elif any(x0 < 0): 
        raise ValueError('x0 is infeasible: negative components.')
    elif len(nbasic) != n - m:
        raise ValueError('Invalid x0: x0 must contain n-m zero elements.')
    if np.linalg.matrix_rank(A[:, basic]) < m:
        warn('The initial basis matrix B is singular!') # (cf. eq. (13.13))
    if any(x0[basic] == 0): 
        warn('The initial basis is degenerate.') #  (cf. Definition 13.1)
    
    x = x0.astype(float)
    
    iterates[:, 0] = x
    bases[:, 0] = basic
    
    k = 0             # Iteration counter
    opt = False       # Optimality flag
    exitflag = np.nan # solver status
    # Loop until optimal, detecting unboundedness, or hitting iteration limit
    while not opt:
    
        # Partition A, x, and c, and determine lambda (la) and sN
        B = A[:, basic]  
        N = A[:, nbasic] 
        xB = x[basic]
        cB = c[basic]
        cN = c[nbasic]
        la = np.linalg.solve(B.T, cB) # lambda: multipl. for Ax = b, B'@la = cB
        sN = cN - N.T @ la # s: multipl. for x >= 0 (sN: "reduced cost" of xN)
        
        fval = c.T @ x
        
        if all(sN >= 0): # optimal point found 
            opt = True
            exitflag = 1
            if report:
                print_report(k, basic, nbasic, x, la, sN, fval, indexing, opt)
            break # trim iterates and bases and return
        
        # Pivot: first determine the variable x_q that will enter the basis
        i_min_sN = np.argmin(sN) # find a negative component of sN (s_q < 0)
        q = nbasic[i_min_sN] # x_q will enter the basis (q is a variable index) 
        d = np.linalg.solve(B, A[:, q]) # solve B*d = A_q for d
        if all(d < 0):
            exitflag = -3
            warn('Problem unbounded. Stopped.')
            break # trim iterates and bases and return
        i_d_g_0, = np.where(d > 0) # Indices of the positive elements in d 
        ratios = xB[i_d_g_0] / d[i_d_g_0]
        x_q_plus = np.min(ratios) # minimum over i | d_i > 0 (Dantzig's rule)
        # p is the minimizing i -- will be removed from basis (NOT x_p!):
        p = i_d_g_0[np.argmin(ratios)] 
        
        # p is a column of B (the matrix). Find corresponding variable index:
        var_leaving_B = basic[p] 
        # The variable entering the basis is x_q. Where in N (the set) is x_q? 
        iN_q, = np.where(nbasic == q)
        
        # Update xB+ and xN+:
        xB_plus = xB - d * x_q_plus
        xN_plus = np.zeros(n - m)
        xN_plus[iN_q] = x_q_plus 
        
        if report:
            print_report(k, basic, nbasic, x, la, sN, fval, indexing, opt, 
                         q, d, x_q_plus, p, var_leaving_B, xB_plus, xN_plus)
        
        k += 1
        
        # Update x with new values:
        x[basic] = xB_plus
        x[nbasic] = xN_plus
        
        # Update basic and nonbasic index sets.  
        # Remove the basic variable corresponding to column p of B (matrix):
        nbasic[iN_q] = basic[p]
        # Add q to the basis:
        basic[p] = q
        
        iterates[:, k] = x
        bases[:, k] = basic
        
        if k >= iterlim - 1: 
            exitflag = 0
            warn('Iteration limit reached. Stopped.')
            break 
    
    # Prepare output: multiplier s and dictionary
    s = np.zeros(n)
    s[nbasic] = sN
    output = {'fval':fval, 'iterates':iterates[:, :k+1], 
              'bases':bases[:, :k+1], 'lambda':la, 's':s, 'exitflag':exitflag}
    
    return x, output


def print_report(iteration, basic, nbasic, x, la, sN, fval, indexing, opt, 
                 q=None, d=None, x_q_plus=None, p=None, var_leaving_B=None, 
                 xB_plus=None, xN_plus=None):    
    hline = '-' * 80
    set_format = '.0f'
    set_pre = '{'
    set_suf = '}'
    vec_format = '7.3f'
    vec_pre = '['
    vec_suf = ']\''
    sep = ', '
    print(hline)
    print(f'  Iteration number: {iteration + indexing} ({indexing} indexing)')
    print('  Basic index set:     ' + 
          vec_str(basic + indexing, set_format, sep, set_pre, set_suf))
    print('  Nonbasic index set:  ' + 
          vec_str(nbasic + indexing, set_format, sep, set_pre, set_suf))
    print('  x_B =    ' + vec_str(x[basic], vec_format, sep, vec_pre, vec_suf))
    print('  x_N =    ' + 
          vec_str(x[nbasic], vec_format, sep, vec_pre, vec_suf))
    print('  lambda = ' + vec_str(la, vec_format, sep, vec_pre, vec_suf))
    print('  s_N =    ' + vec_str(sN, vec_format, sep, vec_pre, vec_suf))
    if opt:
        print('\n  OPTIMAL POINT FOUND')
        print('  x^* =    ' + vec_str(x, vec_format, sep, vec_pre, vec_suf))
        print(f'  c\'x^* = {fval:9.4f}')
        print(hline)
    else:
        print('  x =      ' + vec_str(x, vec_format, sep, vec_pre, vec_suf))
        print(f'  c\'x = {fval:14.8f}')
        print(f'  x_{q + indexing} will enter the basis (q = {q + indexing})')
        print('  d =      ' + vec_str(d, vec_format, sep, vec_pre, vec_suf))
        print((f'  x_q+ = x_{q + indexing}+ = {x_q_plus:7.3f} '
                '\t(value of entering variable/step length)'))
        print((f'  x_{var_leaving_B + indexing} will leave the basis '
               f'(position p = {p + indexing} in the basis)'))
        print('  x_B+ =   ' + 
              vec_str(xB_plus, vec_format, sep, vec_pre, vec_suf) + 
              '\t(current basic vector at new point)')
        print('  x_N+ =   ' + 
              vec_str(xN_plus, vec_format, sep, vec_pre, vec_suf) + 
              ' \t(current nonbasic vector at new point)')
        print(hline + '\n')
        

def vec_str(vector, format_spec, sep=', ', prefix='', suffix=''):
    return prefix + sep.join(format(d, format_spec) for d in vector) + suffix


def phase_I(A, b):
    """Finds a basic feasible starting point x0 for an LP in standard form.
    
    Solves the Phase-I problem for starting the simplex method, as described in
    Nocedal & Wright - Numerical Optimization (2006, 2nd ed.), Springer.
    
    Attempts to solve the LP problem
 
             min e'*x,  subject to A*x + E*z = b, (x,z) >= 0,           (13.40)
 
    in which z is an  m-vector of artificial variables, e is an m-vector of 
    ones, and E is a diagonal matrix with E_jj = 1 if b_j >= 0 and
    E_jj = -1 if b_j < 0. The Phase-I problem starts from the basic feasible 
    point x = 0, z_j = |b_j|, j = 1, 2, ..., m.
    
    If the Phase-I problem terminates with a positive objective-function value
    e'*x, the original LP (13.1) is infeasible. 
    
    Note that using the Phase-I solution x0 as an initial point when solving an
    LP on the standard form (13.1) can be inefficient since the Phase-I basis 
    at the solution can contain components of z. See Nocedal and Wright for 
    details on how to formulate a Phase-II problem that starts from the final
    Phase-I basis. 
    
    Args:
        A: Constraint matrix
        b: Constraint right-hand side
        
    Returns:
        x0: Optimal point x^* (a/the solution to the LP).
        output: Dictionary with keys as specified in the simplex docstring. 
        
    """
    
    m, n = A.shape
    
    # The Phase-I cost e multiplies [x', z']':
    e = np.concatenate((np.zeros(n), np.ones(m)))
    
    # E is a diagonal matrix whose diagonal elements are
    # E_jj = +1 if b_j >= 0, E_jj = −1 if b_j < 0
    E = np.diag(-1 + 2 * (b >= 0))
    
    # The augmented constraint matrix [A, E] that multiplies [x', z']':
    A_phase_I = np.hstack((A, E))
    
    # Initial (trivial basic feasible) point for the Phase-I problem (13.40):
    xz0 = np.concatenate((np.zeros(n), np.abs(b))) # (13.41)
    
    # Solve the Phase-I problem (13.40):
    xz, output = simplex(e, A_phase_I, b, xz0)
    
    x0 = xz[:n]
    
    return x0, output
    