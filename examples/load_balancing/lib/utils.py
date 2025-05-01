import numpy as np


EPS = 1e-6


def fix(sol, currentLocations, averageLoad, shardLoads, epsilon, memory_limit, max_iter=1e4):
    # fix columns
    j_list = np.where(sol.sum(0) == 0)[0]
    sol[:, j_list] = currentLocations[:, j_list]
    sol = sol / sol.sum(0)[None, :]
    loads = sol * shardLoads[None, :]

    # fix rows w/o cost
    loads_movable = loads > 0
    for i in range(loads.shape[0]):
        extra_loads_movable = np.where((currentLocations[i] > 0) & (loads[i] == 0))[0]
        extra_loads_movable = extra_loads_movable[:memory_limit - loads_movable[i].sum()]
        loads_movable[i, extra_loads_movable] = True
    for limit in [(1 - epsilon) * averageLoad, (1 + epsilon) * averageLoad]:
        idx_lh = np.where(loads.sum(1) < limit - EPS)[0]
        idx_rh = np.where(loads.sum(1) > limit + EPS)[0]
        for i_lh in idx_lh:
            for i_rh in idx_rh:
                move_val = loads[i_rh][loads_movable[i_lh]]
                move_val_sum = move_val.sum()
                if move_val_sum < EPS:
                    continue
                move_val *= min(1, (loads[i_rh].sum() - limit) / move_val_sum,
                                (limit - loads[i_lh].sum()) / move_val_sum)
                move_val_sum = move_val.sum()
                if move_val_sum < EPS:
                    continue
                loads[i_lh][loads_movable[i_lh]] += move_val
                loads[i_rh][loads_movable[i_lh]] -= move_val

    # fix rows
    idx = loads.sum(1).argsort()
    while loads[idx[0]].sum() < averageLoad * (1 - epsilon) - EPS and max_iter:
        move_j = loads[idx[-1]].argmax()
        move_val = min(averageLoad * (1 - epsilon) -
                       loads[idx[0]].sum(), loads[idx[-1]].sum() - averageLoad * (1 - epsilon), loads[idx[-1], move_j])
        loads[idx[0], move_j] += move_val
        loads[idx[-1], move_j] -= move_val
        idx = loads.sum(1).argsort()
        max_iter -= 1
    idx = idx[(loads[idx] > EPS).sum(1) <= memory_limit]
    while len(idx) >= 2 and loads[idx[-1]].sum() > averageLoad * (1 + epsilon) + EPS and max_iter:
        move_j = loads[idx[-1]].argmax()
        move_val = min(averageLoad * (1 + epsilon) -
                       loads[idx[0]].sum(), loads[idx[-1]].sum() - averageLoad * (1 + epsilon), loads[idx[-1], move_j])
        loads[idx[0], move_j] += move_val
        loads[idx[-1], move_j] -= move_val
        idx = loads.sum(1).argsort()
        idx = idx[(loads[idx] > EPS).sum(1) <= memory_limit]
        max_iter -= 1
    idx = (loads > EPS).sum(1).argsort()
    while (loads[idx[-1]] > EPS).sum() > memory_limit and max_iter:
        move_j = loads[idx[-1]].argmin()
        move_val = min(averageLoad * (1 + epsilon) - loads[idx[0]].sum(), loads[idx[-1], move_j])
        loads[idx[0], move_j] += move_val
        loads[idx[-1], move_j] -= move_val
        idx = (loads > EPS).sum(1).argsort()
        max_iter -= 1

    # check constraints violation
    warning_list = np.where((loads / shardLoads[None, :]).sum(0) > 1 + EPS)[0].tolist()
    if warning_list:
        print(f'Warning: shards {warning_list} are over-allocated!')
    warning_list = np.where((loads / shardLoads[None, :]).sum(0) < 1 - EPS)[0].tolist()
    if warning_list:
        print(f'Warning: shards {warning_list} are under-allocated!')
    warning_list = np.where(loads.sum(1) / averageLoad < 1 - epsilon - EPS)[0].tolist()
    warning_list += np.where(loads.sum(1) / averageLoad > 1 + epsilon + EPS)[0].tolist()
    if warning_list:
        print(f'Warning: servers {warning_list} are not balanced!')
    warning_list = np.where((loads > EPS).sum(1) > memory_limit)[0].tolist()
    if warning_list:
        print(f'Warning: server {warning_list} exceed the memory limit!')

    return loads / shardLoads[None, :], (loads > EPS).astype(int)
