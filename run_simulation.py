#! /usr/bin/env python3
from simulation import run_simulation

alpha_prefs = 1
alpha_caps = 5
n_positions = 10
n_persons = 100
total_cap = 50
n_simulations = 3
sims = run_simulation(alpha_prefs=alpha_prefs, alpha_caps= alpha_caps, n_positions=n_positions, n_persons= n_persons, total_cap= total_cap, n_sims=n_simulations)
