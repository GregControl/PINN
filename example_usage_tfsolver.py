# -*- coding: utf-8 -*-
"""Untitled96.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BbhTTNyOGhG5weHEWClWYVz41TnF4834

example_usage_tfsolver.py

Demonstrates how to use tfsolver.rk4_integration for a forced ODE system.
"""

!git clone https://github.com/GregControl/PINN.git

import sys
sys.path.append('/content/PINN')  # Adjust the path based on where the file is located

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tfsolver import rk4_integration

@tf.function
def forced_input(time, step_time=50.0, step_magnitude=10.0):
    return tf.where(time >= step_time, step_magnitude, 0.0)

@tf.function
def my_system_func(t, state):
    b1, b2, b3 = 0.5, 0.3, 0.2
    A1, A2, A3 = 1.0, 2.0, 3.0

    h1, h2, h3 = state[0], state[1], state[2]
    qi = forced_input(t, step_time=50.0, step_magnitude=10.0)

    dh1_dt = -b1*tf.sqrt(tf.maximum(h1, 0.0))/A1 + qi/A1
    dh2_dt = -b2*tf.sqrt(tf.maximum(h2, 0.0))/A2 + qi/A2
    dh3_dt = -b3*tf.sqrt(tf.maximum(h3, 0.0))/A3

    return tf.stack([dh1_dt, dh2_dt, dh3_dt])

t_final = 3000
dt = 1.0
time_points = tf.range(0.0, t_final + dt, dt, dtype=tf.float32)
initial_state = tf.constant([1.0, 0.5, 200.0], dtype=tf.float32)

solution = rk4_integration(my_system_func, time_points, initial_state, dt)

solution_np = solution.numpy()
time_np = time_points.numpy()

plt.figure()
for i, label in enumerate(["h1", "h2", "h3"]):
    plt.plot(time_np, solution_np[:, i], label=label)
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.title("RK4 Integration with Step Forcing at t=50s")
plt.legend()
plt.grid(True)
plt.show()