# -*- coding: utf-8 -*-
"""Untitled92.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gIM3eL0bq-oyeDBmEWrCcvE481aMr4DF
"""

"""
tfsolver.py

A small library for integrating ODEs in TensorFlow using 4th-order Runge-Kutta (RK4).
"""

import tensorflow as tf

@tf.function
def rk4_integration(system_func, time_points, initial_state, dt):
    """
    Integrate a user-provided ODE system using RK4 in TensorFlow.

    Args:
        system_func: A Python callable with signature system_func(t, state),
            returning the state derivatives (as a tf.Tensor) at time t.
        time_points: 1D tf.Tensor of shape [N], sorted in ascending order, containing
            the time points at which to evaluate the solution.
        initial_state: 1D tf.Tensor (e.g., shape [state_dim]) containing initial conditions.
        dt: float scalar specifying the uniform time step between consecutive time_points.

    Returns:
        A tf.Tensor of shape (N, state_dim) with the solution at each time in `time_points`.
        The first row is equal to initial_state.
    """
    # Number of time steps
    num_steps = tf.shape(time_points)[0]

    # Prepare a TensorArray to store the solution
    states_ta = tf.TensorArray(dtype=tf.float32, size=num_steps, element_shape=initial_state.shape)
    states_ta = states_ta.write(0, initial_state)

    # Define one RK4 step
    def rk4_step(t, state):
        k1 = system_func(t, state)
        k2 = system_func(t + dt/2, state + k1 * (dt/2))
        k3 = system_func(t + dt/2, state + k2 * (dt/2))
        k4 = system_func(t + dt,   state + k3 * dt)
        return state + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)

    # Loop body for tf.while_loop
    def loop_body(i, current_state, states_ta):
        current_time = time_points[i - 1]
        new_state = rk4_step(current_time, current_state)
        states_ta = states_ta.write(i, new_state)
        return i + 1, new_state, states_ta

    # Loop condition
    def loop_cond(i, current_state, states_ta):
        return i < num_steps

    # Initialize loop
    i0 = tf.constant(1)
    _, _, states_ta = tf.while_loop(loop_cond, loop_body, [i0, initial_state, states_ta])

    # Return the stacked solution
    return states_ta.stack()