import numpy as np

class QuarterCarModel:
    """
    Implements the dynamics of a quarter-car active suspension system.

    This class models the vehicle's suspension using a state-space representation.
    It simulates the system's response to a given road profile and an active
    control force from an actuator. The model is designed to be the core
    component of a reinforcement learning environment.
    """
    def __init__(self, dt=0.001):
        """
        Initializes the quarter-car model with physical parameters.

        The parameters are based on a standard passenger car ('Otomobil') from the
        provided research documents.

        Args:
            dt (float): The simulation time step in seconds.
        """
        # --- Physical Parameters ---
        self.m_s = 20.4   # Sprung mass (kg) - vehicle body
        self.m_u = 15.9    # Unsprung mass (kg) - wheel and axle assembly
        self.k_s = 8799.0 # Suspension spring stiffness (N/m)
        self.k_t = 90000.0# Tire stiffness (N/m)
        self.c_s = 100.0 # Suspension damping coefficient (Ns/m) - this is the passive component

        # --- Actuator Properties (Placeholders) ---
        self.max_force = 100.0 # Maximum actuator force (N)
        self.efficiency = 0.8   # Efficiency for regenerative braking

        # --- Simulation Parameters ---
        self.dt = dt  # Time step (s)

        # --- State Vector: z = [x_s, x_s_dot, x_u, x_u_dot] ---
        # x_s:     Sprung mass displacement (m)
        # x_s_dot: Sprung mass velocity (m/s)
        # x_u:     Unsprung mass displacement (m)
        # x_u_dot: Unsprung mass velocity (m/s)
        self.state = np.zeros(4)

        # Pre-calculate the state-space matrices for efficiency
        self._build_state_space_matrices()

    def _build_state_space_matrices(self):
        """Constructs the A, B, and F matrices for the state-space model."""
        self.A = np.array([
            [0, 1, 0, 0],
            [-self.k_s / self.m_s, -self.c_s / self.m_s, self.k_s / self.m_s, self.c_s / self.m_s],
            [0, 0, 0, 1],
            [self.k_s / self.m_u, self.c_s / self.m_u, -(self.k_s + self.k_t) / self.m_u, -self.c_s / self.m_u]
        ])
        self.B = np.array([0, 1 / self.m_s, 0, -1 / self.m_u])
        self.F = np.array([0, 0, 0, self.k_t / self.m_u])

    def step(self, u, x_g):
        """
        Advances the simulation by one time step.

        Args:
            u (float): The control force from the actuator (N).
            x_g (float): The road profile displacement at the current time (m).

        Returns:
            tuple: A tuple containing key performance metrics:
                - state (np.ndarray): The new state vector.
                - x_s_ddot (float): The sprung mass (body) acceleration (m/s^2).
                - p_regen (float): The potential power regenerated by the actuator (W).
        """
        # 1. Clip the actuator force to its physical limits
        u = np.clip(u, -self.max_force, self.max_force)

        # 2. Calculate state derivatives using the state-space equation: z_dot = Az + Bu + Fx_g
        z_dot = self.A @ self.state + self.B * u + self.F * x_g

        # 3. Integrate using the Euler method to find the next state
        self.state += z_dot * self.dt

        # --- Calculate performance metrics ---
        # Sprung mass acceleration (ride comfort metric)
        x_s_ddot = z_dot[1]

        # Potential regenerated power
        relative_velocity = self.state[1] - self.state[3] # x_s_dot - x_u_dot
        inst_power = -u * relative_velocity
        p_regen = max(0, inst_power) * self.efficiency

        return self.state, x_s_ddot, p_regen

    def reset(self):
        """Resets the model to its initial state."""
        self.state = np.zeros(4)
        return self.state
