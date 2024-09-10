import json

from nhatc import ATCVariable, Coordinator, DynamicSubProblem


def import_system_analysis_json(filepath, verbose=True) -> Coordinator:
    with open(filepath) as file:
        analysis_object = json.load(file)

    coordinator = Coordinator(verbose=verbose)
    v_idx_to_av_map = {}
    v_idx_intermediate_set = set()
    v_exp_table = {}
    v_list = []
    intermediate_variables = {} # Maps from sub system index (ss_idx) to array of

    for variable in analysis_object['variables']:
        v_idx = variable['v_index']
        v_type = variable['type']
        ss_idx = variable['ss_index']

        if ss_idx not in intermediate_variables:
            intermediate_variables[ss_idx] = []

        if v_type == 'intermediate':
            intermediate_variables[ss_idx].append(variable)
            v_idx_intermediate_set.add(v_idx)
            continue

        name = variable['symbol']
        links = variable['links']
        lb = variable['lb']
        ub = variable['ub']
        v_exp_table[v_idx] = variable['expression']

        coupling = True if v_type == 'coupling' else False
        av = ATCVariable(name, v_idx, ss_idx, coupling, links, lb, ub)
        v_list.append(av)
        v_idx_to_av_map[v_idx] = av

        if verbose:
            print(av)

    coordinator.set_variables(v_list)

    sp_list = []
    for subsystem in analysis_object['subsystems']:
        sp = DynamicSubProblem()
        sp.index = subsystem['ss_index']
        sp.obj = str(subsystem['objective'])

        sp_vars = {}
        sp_couplings = {}
        for v_idx in subsystem['variables']:
            # Add optimization variables to sub-system

            if v_idx in v_idx_intermediate_set:
                # Intermediate variables are not optimization variables
                continue

            av = v_idx_to_av_map[v_idx]
            if av.coupled_variable:
                sp_couplings[av.name] = v_exp_table[v_idx]
            else:
                sp_vars[av.name] = v_idx

        sp.variables = sp_vars
        sp.couplings = sp_couplings

        # Register intermediate variable expressions in sub-system
        for var_inter in intermediate_variables[sp.index]:
            print(f'Added {var_inter} to SP{sp.index}')
            sp.add_intermediate_variable(var_inter['symbol'], var_inter['expression'])

        for constraint in subsystem['constraints']:
            if constraint['type'] == 'ieq':
                sp.inequality_constraints.append(constraint['expression'])
            elif constraint['type'] == 'eq':
                sp.equality_constraints.append(constraint['expression'])
            else:
                raise ValueError(f'Unknown constraint type' + str({constraint['type']}))

        sp_list.append(sp)

    coordinator.set_subproblems(sp_list)
    return coordinator
