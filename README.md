# Install
```pip install nhatc```

# Non-Hierarchical Analytical Target Cascading
This library is an interpretation of Non-hierarchical analytical target cascading,
as specified by Talgorn and Kokkolaras in their 2017 paper 
(doi.org/10.1007/s00158-017-1726-0).

The primary objective of this library is to make NHATC as simple as possible to use. 
Specifically, the main focus is to simplify integration of NHATC into other 
software using the "dynamic approach" demonstrated in the usage examples below. 
This approach enables NHATC to be integrated more easily into GUIs, as the sub-systems
can be defined "dynamically" without the use of functions.

This approach relies on three core technologies: 
- **numpy** for handling all vectors
- **scipy** for objective optimization
- **cexprtk** for parsing and evaluating user-defined mathematical expressions (dynamic approach only)

# Usage
The library can either be used programmatically. For instance, you may be using jupyter to 
set up and configure your optimization problem. Or, you can use the library dynamically, which is better suited for
when the input is non-static: i.e., when the input is controlled through a GUI.


## Programmatic example

```python
from nhatc import ATCVariable, Coordinator, ProgrammaticSubProblem

coordinator = Coordinator(verbose=True) # Verbose: print process to terminal
# Define variables. The constructor arguments for the ATCVariable class are:
# Variable name, Variable index, Sub-system index, Coupling variable, Lower bound, Upper bound
coordinator.set_variables([
    ATCVariable('a1', 0, 0, True, [3], 0, 10),
    ATCVariable('b1', 1, 0, False, [4], 0, 10),
    ATCVariable('w1', 2, 0, False, [5], 0, 10),
    ATCVariable('a2', 3, 1, False, [0], 0, 10),
    ATCVariable('b2', 4, 1, True, [1], 0, 10),
    ATCVariable('w2', 5, 1, False, [2], 0, 10),
])

# Define sub-systems as functions. 
# Sub-system functions output coupled variables (y) AND the objective (f)
# The variables have the same indices in X as defined above in the coordinator variable list 
def sp1_objective(X):
    b, w = X[[1, 2]]
    a = w + (1/b**2)
    f = (a + b) / w
    y = [a]
    return f, y


def sp2_objective(X):
    a, w = X[[3, 5]]
    b = (a/2) * w
    y = [b]
    f = 0
    return f, y

# Any constraints are defined separately as functions. 
# Note the g(x) ≥ 0 formulation, which is the opposite of what is typically used in Matlab
def sp2_ineq(X):
    # g(x) ≥ 0
    b, w = X[[4, 5]]
    return 3 - (b + w)  # 3 - ( b + w ) ≥ 0


sp1 = ProgrammaticSubProblem(0)
sp1.set_objective(sp1_objective)
sp2 = ProgrammaticSubProblem(1)
sp2.set_objective(sp2_objective)
# Add constraints to sub-problem
sp2.set_ineqs([sp2_ineq])

coordinator.set_subproblems([sp1, sp2])

# Generate random x0 based on variable definitions above
x0 = coordinator.get_random_x0()
# Run optimization coordination.
res = coordinator.optimize(100, x0,
                           beta=2.0,    # Penalty update parameter 1
                           gamma=0.25,  # Penalty update parameter 2
                           convergence_threshold=1e-9, 
                           method='slsqp')  # Optimization method

if res:
    if res.successful_convergence:
        print(f'Reached convergence')
    else:
        print(f'FAILED to reach convergence')

    print(f'Process time: {res.time} seconds')
    print("Verification against objectives:")
    print(f'f* = {res.f_star[0]}')
    print(f'Epsilon = {res.epsilon} ')

    print('x*:')
    for i, x_i in enumerate(res.x_star):
        name = coordinator.variables[i].name
        lb = coordinator.variables[i].lb
        ub = coordinator.variables[i].ub

        print(f'{name}\t[{lb}; {ub}]\tvalue: {x_i}')
else:
    print('Optimization failed')

```

## Dynamic example

```python
from nhatc import ATCVariable, Coordinator, DynamicSubProblem

# Instantiate coordinator and variables
coordinator = Coordinator(verbose=True)     # Verbose -> outputs progress
coordinator.set_variables([
    ATCVariable('a1', 0, 0, True, [3], 0, 10),
    ATCVariable('b1', 1, 0, False, [4], 0, 10),
    ATCVariable('w1', 2, 0, False, [5], 0, 10),
    ATCVariable('a2', 3, 1, False, [0], 0, 10),
    ATCVariable('b2', 4, 1, True, [1], 0, 10),
    ATCVariable('w2', 5, 1, False, [2], 0, 10),
])

# Setup sub-problem 1
spi_1 = DynamicSubProblem()
# Sub problem index which are references in the aforementioned coordinator variables 
spi_1.index = 0
# Configure mathematical scope of the sub-problem
spi_1.variables = {'b': 1, 'w': 2}
# Define variables that are coupled with other sub-problems using previously defined variables
spi_1.couplings = {'a': 'w + (1/(b^2))'}
# Define the main objective function
spi_1.obj = "(a + b) / w"

# Setup sub-problem 2
spi_2 = DynamicSubProblem()
spi_2.index = 1
# In ATC, there typically only exists one objective function. 
# Consequently, the objectives of all other sub-systems are normally "0"
spi_2.obj = "0" 
spi_2.variables = {'a': 3, 'w': 5}
spi_2.couplings = {'b': '(a/2) * w'}
spi_2.inequality_constraints.append('3 - ( b + w )')

# Add sub-problems to coordinator
coordinator.set_subproblems([spi_1, spi_2])

# Get a random starting point based on bounds defined in coordinator variables
x0 = coordinator.get_random_x0()
# Run optimization algorithm. 
# You can choose method (e.g., slsqp, nelder-mead, ...). 
# This utilizes the minimize function from scipy, 
# so any methods that works in minimize also works here.
res = coordinator.optimize(100, x0,
                           beta=2.0,
                           gamma=0.25,
                           convergence_threshold=1e-9,
                           NI=60,
                           method='slsqp')

# Manage the results
if res:
    if res.successful_convergence:
        print(f'Reached convergence')
    else:
        print(f'FAILED to reach convergence')

    print(f'Process time: {res.time} seconds')
    print("Verification against objectives:")
    print(f'f* = {res.f_star[0]}')
    print(f'Epsilon = {res.epsilon} ')

    print('x*:')
    for i, x_i in enumerate(res.x_star):
        name = coordinator.variables[i].name
        lb = coordinator.variables[i].lb
        ub = coordinator.variables[i].ub

        print(f'{name}\t[{lb}; {ub}]\tvalue: {x_i}')
else:
    print('Optimization failed')
```

