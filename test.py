from nhatc.utils import import_system_analysis_json
import cexprtk

filepath = r'C:\Users\julianm\OneDrive - Chalmers\Documents\projects\nhatc\system_analysis.json'

coordinator = import_system_analysis_json(filepath)
x0 = coordinator.get_random_x0()
res = coordinator.optimize(100, x0,
                           beta=2.0,
                           gamma=0.25,
                           convergence_threshold=1e-9,
                           NI=60,
                           method='slsqp')

if res:
    if res.successful_convergence:
        print(f'Reached convergence')
    else:
        print(f'FAILED to reach convergence')

    print(f'Process time: {res.time} seconds')
    print("Verification against objectives:")
    print(f'f* = {res.f_star[0]}')
    print(f'Epsilon = {res.epsilon} ')
    print(res.f_star)

    print('x*:')
    for i, x_i in enumerate(res.x_star):
        name = coordinator.variables[i].name
        lb = coordinator.variables[i].lb
        ub = coordinator.variables[i].ub

        print(f'{name}\t[{lb}; {ub}]\tvalue: {x_i}')
else:
    print('Catastrophic failure')
