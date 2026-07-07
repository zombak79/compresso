def exponential_decay(n_init, n_final, k):
    """
    Smooth exponential decay from n_init to n_final across k steps,
    guaranteed never to go below n_final.
    """
    if k <= 0:
        return [n_init]
    r = (n_final / n_init) ** (1 / k)
    values = [n_init]
    for _ in range(k):
        next_val = int(values[-1] * r)
        if next_val <= n_final:
            return values
        values.append(next_val)
    return values