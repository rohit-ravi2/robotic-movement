"""Flight scoring rule shared by both simulators (no pydrake import)."""
import numpy as np
FLIGHT_CONFIRM = 0.020

def score(t, vz, fz):
    """Best confirmed flight: (v_takeoff, t_takeoff, n_flights)."""
    best = (0.0, None); n = 0; i = 0; N = len(t)
    while i < N:
        if fz[i] < 1.0 and vz[i] > 0.0 and i > 0:
            j = i
            while j < N and fz[j] < 1.0: j += 1
            if (t[min(j, N-1)] - t[i]) >= FLIGHT_CONFIRM - 1e-12 or j == N:
                n += 1
                if vz[i] > best[0]: best = (float(vz[i]), float(t[i]))
            i = j
        else:
            i += 1
    return best[0], best[1], n

