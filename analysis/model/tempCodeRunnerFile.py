    idx = step_ts.index.to_numpy()
    vals = step_ts.to_numpy()
    if target_step <= idx.min():
        return vals[0]
    if target_step >= idx.max():
        return vals[-1]
    pos = np.searchsorted(idx, target_step, side="left")
    return vals[pos]


