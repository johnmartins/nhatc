import json

from nhatc import ATCVariable, Coordinator, DynamicSubProblem


def import_system_analysis_json(filepath, verbose=True) -> Coordinator:
    with open(filepath) as file:
        analysis_object = json.load(file)

    coordinator = Coordinator(verbose=verbose)
    av_idx_map = {}
    v_exp_table = {}
    v_list = []

    for variable in analysis_object['variables']:
        v_idx = variable['v_index']
        ss_idx = variable['ss_index']
        name = variable['symbol']
        type = variable['type']
        links = variable['links']
        lb = variable['lb']
        ub = variable['ub']
        v_exp_table[v_idx] = variable['expression']

        coupling = True if type == 'coupling' else False
        av = ATCVariable(name, v_idx, ss_idx, coupling, links, lb, ub)
        v_list.append(av)
        av_idx_map[v_idx] = av

    coordinator.set_variables(v_list)

    sp_list = []
    for subsystem in analysis_object['subsystems']:
        sp = DynamicSubProblem()
        sp.index = subsystem['ss_index']
        sp.obj = subsystem['objective']

        sp_vars = {}
        sp_couplings = {}
        print(subsystem['variables'])
        for v_idx in subsystem['variables']:
            av = av_idx_map[v_idx]
            if av.coupled_variable:
                sp_couplings[av.name] = v_exp_table[v_idx]
            else:
                sp_vars[av.name] = v_idx

        sp.variables = sp_vars
        sp.couplings = sp_couplings

        print(sp.variables)

        sp_list.append(sp)

    coordinator.set_subproblems(sp_list)
    return coordinator
