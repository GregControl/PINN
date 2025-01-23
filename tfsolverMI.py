# tfsolverMI.py
import tensorflow as tf

@tf.function
def rk4_integration(system_func, 
                    time_points, 
                    initial_state, 
                    dt,
                    forced_data=None):
    """
    Integrate a user-provided ODE system using 4th-order Runge-Kutta (RK4) in TensorFlow.
    Optionally handle time-varying forced inputs.

    Args:
        system_func: A Python callable with signature:
                     system_func(t, state, input_vec) -> tf.Tensor
          - t:        scalar tf.float32, the current time
          - state:    1D tf.Tensor (shape [state_dim]) with the current state
          - input_vec: 1D tf.Tensor (shape [num_inputs]) if forced_data is provided
                       or None if no forced_data is used
          - returns:  1D tf.Tensor (shape [state_dim]) representing d(state)/dt

        time_points:   1D tf.Tensor of shape [N], sorted in ascending order,
                       containing the times at which to evaluate the solution.
        initial_state: 1D tf.Tensor (shape [state_dim]) containing initial conditions.
        dt:            float scalar specifying the uniform time step between consecutive time_points.
        forced_data:   Optional. A 2D tf.Tensor of shape [N, M] containing external/forced inputs 
                       at each time step. 
                       - If provided, forced_data[i] is the input vector at time_points[i]. 
                       - If None, system_func will be called with input_vec=None.

    Returns:
        A tf.Tensor of shape (N, state_dim) with the solution at each time in `time_points`.
        The first row is equal to `initial_state`.
    """
    # Number of time steps
    num_steps = tf.shape(time_points)[0]

    # Prepare a TensorArray to store the solution
    states_ta = tf.TensorArray(dtype=tf.float32, 
                               size=num_steps, 
                               element_shape=initial_state.shape)
    states_ta = states_ta.write(0, initial_state)

    @tf.function
    def rk4_step(t, state, input_vec):
        """
        One step of 4th-order Runge-Kutta, from time t to t+dt.

        system_func(t, state, input_vec) should return d(state)/dt.
        input_vec is the forced input (or None) for the time interval [t, t+dt).
        """
        k1 = system_func(t,         state,              input_vec)
        k2 = system_func(t + dt/2,  state + k1*(dt/2),  input_vec)
        k3 = system_func(t + dt/2,  state + k2*(dt/2),  input_vec)
        k4 = system_func(t + dt,    state + k3*dt,      input_vec)
        return state + (k1 + 2*k2 + 2*k3 + k4)*(dt/6.0)

    def loop_body(i, current_state, states_ta):
        # The time at the previous step is time_points[i-1].
        current_time = time_points[i - 1]

        if forced_data is not None:
            # forced_data[i-1] is the input vector for the interval [i-1 -> i]
            input_vec = forced_data[i - 1]
        else:
            input_vec = None

        new_state = rk4_step(current_time, current_state, input_vec)
        states_ta = states_ta.write(i, new_state)

        return i+1, new_state, states_ta

    def loop_cond(i, current_state, states_ta):
        return i < num_steps

    # Initialize tf.while_loop
    i0 = tf.constant(1)
    _, _, states_ta = tf.while_loop(loop_cond, loop_body, [i0, initial_state, states_ta])

    return states_ta.stack()
