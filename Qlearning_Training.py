import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from Suspension_Model import QuarterCarModel
from Road_profile import SquareWaveProfile, BumpProfile

class QuadraticQLearning:
    """
    Implements Quadratic Q-Learning for Active Suspension Control
    as described in the research project.
    
    The Q-function has the form: Q(z) = z^T @ Q_matrix @ z
    where z = [x_s, x_s_dot, x_u, x_u_dot, u, x_g]
    """
    
    def __init__(self, 
                 state_dim=4,
                 action_dim=1, 
                 disturbance_dim=1,
                 learning_rate=0.01,
                 gamma=0.95,
                 exploration_noise=0.1):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.disturbance_dim = disturbance_dim
        self.total_dim = state_dim + action_dim + disturbance_dim  # 6D
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_noise = exploration_noise
        
        # Initialize Q-matrix with proper H∞ block structure
        self._initialize_hinf_q_matrix()
        
        # Optimizer for Q-matrix
        self.optimizer = optim.Adam([self.Q_matrix], lr=learning_rate)
        
        # Multi-objective weights (can be adjusted)
        self.weight_comfort = 0.4    # a1 for Jk (comfort)
        self.weight_energy = 0.3     # a2 for Je (energy)  
        self.weight_handling = 0.3   # a3 for Jg (handling)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

    def _initialize_hinf_q_matrix(self):
        """
        Initialize Q-matrix with proper H∞ control structure for zero-sum game.
        
        Structure: z = [x_s, x_s_dot, x_u, x_u_dot, u, x_g]
                     [    State (4D)          C  D ]
        
        Q = [ Q_xx  Q_xu  Q_xw ]  where:
            [ Q_xu' Q_uu  Q_uw ]  - Q_xx: State penalty matrix (4×4)
            [ Q_xw' Q_uw' Q_ww ]  - Q_uu: Control penalty (negative for maximization)
                                  - Q_ww: Disturbance penalty (positive for minimization)
        """
        self.Q_matrix = torch.zeros(self.total_dim, self.total_dim, dtype=torch.float32)
        
        # Q_xx: State penalty matrix (4×4) - based on performance criteria
        # Penalize large displacements and velocities
        Q_xx = torch.diag(torch.tensor([1.0, 0.1, 1.0, 0.1]))  # [x_s, x_s_dot, x_u, x_u_dot]
        self.Q_matrix[:4, :4] = Q_xx
        
        # Q_uu: Control penalty (scalar) - NEGATIVE for maximization in zero-sum game
        self.Q_matrix[4, 4] = -0.01  # Negative because controller maximizes
        
        # Q_ww: Disturbance penalty (scalar) - POSITIVE for minimization
        self.Q_matrix[5, 5] = 0.1   # Positive because disturbance minimizes
        
        # Q_xu: State-control coupling (4×1) - small random initialization
        self.Q_matrix[:4, 4] = torch.randn(4) * 0.01
        self.Q_matrix[4, :4] = self.Q_matrix[:4, 4]  # Ensure symmetry
        
        # Q_xw: State-disturbance coupling (4×1) - small random initialization  
        self.Q_matrix[:4, 5] = torch.randn(4) * 0.01
        self.Q_matrix[5, :4] = self.Q_matrix[:4, 5]  # Ensure symmetry
        
        # Q_uw: Control-disturbance coupling (scalar)
        self.Q_matrix[4, 5] = torch.randn(1) * 0.01
        self.Q_matrix[5, 4] = self.Q_matrix[4, 5]  # Ensure symmetry
        
        self.Q_matrix.requires_grad_(True)

    def compute_q_value(self, z):
        """
        Compute Q-value for augmented state-action-disturbance vector z
        Q(z) = z^T @ Q_matrix @ z
        """
        if isinstance(z, np.ndarray):
            z = torch.FloatTensor(z)
        
        return torch.sum(z * (self.Q_matrix @ z), dim=-1)
    
    def get_optimal_action(self, state, disturbance):
        """
        Solve for optimal action using H∞ saddle point: max_u min_w Q(x,u,w)
        
        For the quadratic form Q(z) = z^T @ Q @ z, the optimal control is:
        u* = -Q_uu^(-1) * (Q_xu^T * x + Q_uw * w)
        
        This assumes Q_uu < 0 (since controller maximizes Q)
        """
        state = torch.FloatTensor(state) if isinstance(state, np.ndarray) else state
        disturbance = torch.FloatTensor([disturbance]) if isinstance(disturbance, (int, float)) else disturbance
        
        # Extract blocks from Q-matrix
        # Q_matrix structure: [x_s, x_s_dot, x_u, x_u_dot, u, x_g]
        # Indices:            [ 0,    1,      2,    3,     4, 5 ]
        
        Q_xu = self.Q_matrix[:4, 4]      # State-control coupling (4×1)
        Q_uu = self.Q_matrix[4, 4]       # Control penalty (scalar)
        Q_uw = self.Q_matrix[4, 5]       # Control-disturbance coupling (scalar)
        
        # Compute optimal action using H∞ formula
        # u* = -Q_uu^(-1) * (Q_xu^T * x + Q_uw * w)
        if abs(Q_uu) > 1e-6:  # Avoid division by zero
            # Note: Q_uu should be negative for maximization, so this gives positive feedback
            optimal_action = -(Q_xu @ state + Q_uw * disturbance) / Q_uu
        else:
            # Fallback to zero if Q_uu is too small
            optimal_action = torch.tensor(0.0)
            
        return optimal_action.item()
    
    def get_action_with_exploration(self, state, disturbance):
        """Get action with exploration noise for training"""
        optimal_action = self.get_optimal_action(state, disturbance)
        
        # Add exploration noise
        noise = np.random.normal(0, self.exploration_noise)
        action = optimal_action + noise
        
        # Clip to actuator limits (from suspension model)
        action = np.clip(action, -100.0, 100.0)
        
        return action
    
    def compute_reward(self, state, action, next_state, p_regen, x_s_ddot):
        """
        Compute multi-objective reward: J = a1*Jk - a2*Je + a3*Jg
        """
        # Jk: Comfort index (minimize acceleration)
        comfort_cost = x_s_ddot**2
        
        # Je: Energy index (maximize regeneration, minimize consumption)
        power_consumed = abs(action * (state[1] - state[3]))  # |F * v_rel|
        energy_reward = p_regen - 0.1 * power_consumed
        
        # Jg: Handling index (minimize tire deflection)
        tire_deflection = abs(state[2])  # |x_u| 
        handling_cost = tire_deflection**2
        
        # Combined reward
        reward = (self.weight_comfort * (-comfort_cost) + 
                 self.weight_energy * energy_reward + 
                 self.weight_handling * (-handling_cost))
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, disturbance, next_disturbance):
        """Store experience in replay buffer"""
        self.memory.append((state.copy(), action, reward, next_state.copy(), 
                          disturbance, next_disturbance))
    
    def update_q_matrix(self):
        """Update Q-matrix using batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch_experiences = [self.memory[i] for i in batch]
        
        total_loss = 0
        
        for state, action, reward, next_state, disturbance, next_disturbance in batch_experiences:
            # Current Q-value
            z_current = torch.FloatTensor(np.concatenate([state, [action], [disturbance]]))
            q_current = self.compute_q_value(z_current)
            
            # Next Q-value (with optimal next action)
            next_action = self.get_optimal_action(next_state, next_disturbance)
            z_next = torch.FloatTensor(np.concatenate([next_state, [next_action], [next_disturbance]]))
            q_next = self.compute_q_value(z_next)
            
            # Target Q-value
            q_target = reward + self.gamma * q_next
            
            # Loss (TD error)
            loss = (q_current - q_target.detach())**2
            total_loss += loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Maintain H∞ structure constraints
        self._enforce_hinf_structure()

    def _enforce_hinf_structure(self):
        """
        Enforce H∞ structure constraints after each update:
        1. Maintain symmetry
        2. Keep Q_uu negative (for maximization)
        3. Keep Q_ww positive (for minimization)
        """
        with torch.no_grad():
            # Ensure symmetry
            self.Q_matrix.data = (self.Q_matrix.data + self.Q_matrix.data.T) / 2
            
            # Enforce game-theoretic structure
            # Q_uu should be negative (controller maximizes)
            if self.Q_matrix[4, 4] > -1e-6:
                self.Q_matrix[4, 4] = -1e-6
                
            # Q_ww should be positive (disturbance minimizes)  
            if self.Q_matrix[5, 5] < 1e-6:
                self.Q_matrix[5, 5] = 1e-6

    def save_agent(self, filepath):
        """Save the trained Q-matrix and agent parameters"""
        save_dict = {
            'Q_matrix': self.Q_matrix.detach().numpy(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'disturbance_dim': self.disturbance_dim,
            'weights': {
                'comfort': self.weight_comfort,
                'energy': self.weight_energy,
                'handling': self.weight_handling
            },
            'hyperparams': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'exploration_noise': self.exploration_noise
            }
        }
        np.save(filepath, save_dict)
        print(f"Agent saved to {filepath}")
    
    def load_agent(self, filepath):
        """Load a trained Q-matrix and agent parameters"""
        save_dict = np.load(filepath, allow_pickle=True).item()
        
        self.Q_matrix = torch.FloatTensor(save_dict['Q_matrix'])
        self.Q_matrix.requires_grad_(True)
        
        # Update optimizer with loaded Q-matrix
        self.optimizer = optim.Adam([self.Q_matrix], lr=self.learning_rate)
        
        # Load weights and hyperparams
        weights = save_dict['weights']
        self.weight_comfort = weights['comfort']
        self.weight_energy = weights['energy'] 
        self.weight_handling = weights['handling']
        
        print(f"Agent loaded from {filepath}")
    
    def get_training_info(self):
        """Get current training state information"""
        info = {
            'Q_matrix_norm': torch.norm(self.Q_matrix).item(),
            'Q_uu': self.Q_matrix[4,4].item(),
            'Q_ww': self.Q_matrix[5,5].item(),
            'exploration_noise': self.exploration_noise,
            'memory_size': len(self.memory)
        }
        return info

class SuspensionEnvironment:
    """Environment wrapper for the suspension system"""
    
    def __init__(self, dt=0.001, episode_length=5.0):
        self.suspension = QuarterCarModel(dt=dt)
        self.road_profile = SquareWaveProfile(period=2.0, amplitude=0.02)
        
        self.dt = dt
        self.episode_length = episode_length
        self.max_steps = int(episode_length / dt)
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.suspension.reset()
        self.current_step = 0
        self.current_time = 0.0
        return self.suspension.state
    
    def step(self, action):
        """Take one step in the environment"""
        # Get current road disturbance
        road_input = self.road_profile.get_profile(self.current_time)
        
        # Step the suspension model
        next_state, x_s_ddot, p_regen = self.suspension.step(action, road_input)
        
        # Update time
        self.current_time += self.dt
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get next road input for next step
        next_road_input = self.road_profile.get_profile(self.current_time)
        
        return next_state, x_s_ddot, p_regen, road_input, next_road_input, done

def train_quadratic_q_learning(episodes=1000, 
                              save_path="trained_suspension_agent.npy",
                              convergence_window=100,
                              min_improvement=0.001,
                              verbose=True):
    """
    Main training loop with saving and convergence monitoring
    
    Args:
        episodes (int): Maximum number of episodes to train
        save_path (str): Where to save the trained agent
        convergence_window (int): Window to check for convergence
        min_improvement (float): Minimum improvement to continue training
        verbose (bool): Print training progress
    
    Returns:
        tuple: (trained_agent, training_metrics)
    """
    
    # Initialize environment and agent
    env = SuspensionEnvironment()
    agent = QuadraticQLearning()
    
    # Training metrics
    episode_rewards = []
    comfort_metrics = []
    energy_metrics = []
    q_matrix_norms = []
    
    print(f"🚀 Starting Training...")
    print(f"📊 Episodes: {episodes}")
    print(f"⏱️  Episode Length: {env.episode_length}s ({env.max_steps} steps)")
    print(f"🎯 Multi-objective weights: Comfort={agent.weight_comfort}, Energy={agent.weight_energy}, Handling={agent.weight_handling}")
    print("-" * 50)
    
    best_avg_reward = float('-inf')
    episodes_without_improvement = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_comfort = 0
        episode_energy = 0
        step_count = 0
        
        # Get initial road input
        road_input = env.road_profile.get_profile(0.0)
        
        # Episode simulation loop
        while True:
            # Get action from agent
            action = agent.get_action_with_exploration(state, road_input)
            
            # Take step in environment (calls YOUR suspension model)
            next_state, x_s_ddot, p_regen, current_road, next_road, done = env.step(action)
            
            # Compute multi-objective reward
            reward = agent.compute_reward(state, action, next_state, p_regen, x_s_ddot)
            
            # Store experience for Q-matrix learning
            agent.store_experience(state, action, reward, next_state, current_road, next_road)
            
            # Update metrics
            episode_reward += reward
            episode_comfort += x_s_ddot**2
            episode_energy += p_regen
            step_count += 1
            
            # Update state
            state = next_state
            road_input = next_road
            
            if done:
                break
        
        # Update Q-matrix using experience replay
        agent.update_q_matrix()
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        comfort_metrics.append(episode_comfort / step_count)  # Average per step
        energy_metrics.append(episode_energy)
        
        # Track Q-matrix evolution
        training_info = agent.get_training_info()
        q_matrix_norms.append(training_info['Q_matrix_norm'])
        
        # Decay exploration noise
        agent.exploration_noise = max(0.01, agent.exploration_noise * 0.995)
        
        # Check for convergence
        if episode >= convergence_window:
            recent_rewards = episode_rewards[-convergence_window:]
            avg_reward = np.mean(recent_rewards)
            
            if avg_reward > best_avg_reward + min_improvement:
                best_avg_reward = avg_reward
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 1
        
        # Verbose output
        if verbose and episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode >= 100 else np.mean(episode_rewards)
            avg_comfort = np.mean(comfort_metrics[-100:]) if episode >= 100 else np.mean(comfort_metrics)
            avg_energy = np.mean(energy_metrics[-100:]) if episode >= 100 else np.mean(energy_metrics)
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {avg_reward:8.3f} | "
                  f"Comfort: {avg_comfort:8.3f} | "
                  f"Energy: {avg_energy:6.2f} | "
                  f"Q_uu: {training_info['Q_uu']:6.3f} | "
                  f"Noise: {agent.exploration_noise:.3f}")
        
        # Early stopping if converged
        if episodes_without_improvement >= convergence_window * 2:
            print(f"\n✅ Converged after {episode} episodes!")
            break
    
    # Save the trained agent
    agent.save_agent(save_path)
    
    # Final training summary
    print(f"\n🎉 Training Complete!")
    print(f"📁 Agent saved to: {save_path}")
    print(f"📈 Final average reward: {np.mean(episode_rewards[-100:]):.3f}")
    print(f"🛡️  Final Q_uu (control penalty): {training_info['Q_uu']:.6f}")
    print(f"⚡ Final Q_ww (disturbance penalty): {training_info['Q_ww']:.6f}")
    
    # Package training results
    training_metrics = {
        'episode_rewards': episode_rewards,
        'comfort_metrics': comfort_metrics,
        'energy_metrics': energy_metrics,
        'q_matrix_norms': q_matrix_norms,
        'final_q_matrix': agent.Q_matrix.detach().numpy(),
        'training_info': training_info
    }
    
    return agent, training_metrics

def test_trained_agent(agent_path="trained_suspension_agent.npy", test_duration=10.0):
    """
    Test a trained agent on different road profiles
    """
    # Load trained agent
    agent = QuadraticQLearning()
    agent.load_agent(agent_path)
    agent.exploration_noise = 0.0  # No exploration during testing
    
    # Test environment
    env = SuspensionEnvironment(episode_length=test_duration)
    
    # Test on different road profiles
    test_profiles = [
        ("Square Wave", SquareWaveProfile(period=2.0, amplitude=0.02)),
        ("Bump", BumpProfile(start_time=2.0, duration=1.0, height=0.05)),
    ]
    
    results = {}
    
    for profile_name, road_profile in test_profiles:
        env.road_profile = road_profile
        state = env.reset()
        
        # Track performance metrics
        comfort_score = 0
        energy_recovered = 0
        handling_score = 0
        step_count = 0
        
        states_history = []
        actions_history = []
        road_history = []
        
        road_input = env.road_profile.get_profile(0.0)
        
        while True:
            # Get action (no exploration)
            action = agent.get_optimal_action(state, road_input)
            
            # Step environment
            next_state, x_s_ddot, p_regen, current_road, next_road, done = env.step(action)
            
            # Update metrics
            comfort_score += x_s_ddot**2
            energy_recovered += p_regen
            handling_score += abs(state[2])**2  # tire deflection
            step_count += 1
            
            # Store for plotting
            states_history.append(state.copy())
            actions_history.append(action)
            road_history.append(current_road)
            
            state = next_state
            road_input = next_road
            
            if done:
                break
        
        # Store results
        results[profile_name] = {
            'comfort_score': comfort_score / step_count,
            'energy_recovered': energy_recovered,
            'handling_score': handling_score / step_count,
            'states': np.array(states_history),
            'actions': np.array(actions_history),
            'road_inputs': np.array(road_history),
            'time': np.arange(step_count) * env.dt
        }
        
        print(f"{profile_name} Test Results:")
        print(f"  Comfort Score (lower=better): {results[profile_name]['comfort_score']:.6f}")
        print(f"  Energy Recovered: {results[profile_name]['energy_recovered']:.3f} W")
        print(f"  Handling Score (lower=better): {results[profile_name]['handling_score']:.6f}")
        print()
    
    return results

# Example usage
if __name__ == "__main__":
    print("🔧 Quadratic Q-Learning for Active Suspension Control")
    print("=" * 50)
    
    # Training
    print("Starting training...")
    trained_agent, training_metrics = train_quadratic_q_learning(
        episodes=500,
        save_path="suspension_agent.npy",
        verbose=True
    )
    
    # Plot training results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(training_metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(training_metrics['comfort_metrics'])
    plt.title('Comfort Metric (Lower is Better)')
    plt.xlabel('Episode')
    plt.ylabel('Average Acceleration²')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(training_metrics['energy_metrics'])
    plt.title('Energy Recovery')
    plt.xlabel('Episode')
    plt.ylabel('Total Regenerated Power (W)')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(training_metrics['q_matrix_norms'])
    plt.title('Q-Matrix Evolution')
    plt.xlabel('Episode')
    plt.ylabel('Q-Matrix Frobenius Norm')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    final_q = training_metrics['final_q_matrix']
    plt.imshow(final_q, cmap='RdBu_r', vmin=-np.max(np.abs(final_q)), vmax=np.max(np.abs(final_q)))
    plt.title('Final Q-Matrix Structure')
    plt.colorbar()
    labels = ['x_s', 'x_s_dot', 'x_u', 'x_u_dot', 'u', 'x_g']
    plt.xticks(range(6), labels)
    plt.yticks(range(6), labels)
    
    plt.subplot(2, 3, 6)
    # Show Q_uu and Q_ww evolution to verify game structure
    q_uu_history = [training_metrics['final_q_matrix'][4,4]] * len(training_metrics['episode_rewards'])
    q_ww_history = [training_metrics['final_q_matrix'][5,5]] * len(training_metrics['episode_rewards'])
    plt.plot(q_uu_history, label='Q_uu (should be negative)', color='red')
    plt.plot(q_ww_history, label='Q_ww (should be positive)', color='blue')
    plt.title('Game Structure Parameters')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Testing
    print("\n🧪 Testing trained agent...")
    test_results = test_trained_agent("suspension_agent.npy", test_duration=5.0)
    
    # Plot test results
    plt.figure(figsize=(15, 8))
    
    for i, (profile_name, data) in enumerate(test_results.items()):
        plt.subplot(2, len(test_results), i + 1)
        plt.plot(data['time'], data['states'][:, 0], label='x_s (sprung mass)')
        plt.plot(data['time'], data['states'][:, 2], label='x_u (unsprung mass)')
        plt.plot(data['time'], data['road_inputs'], label='Road input', alpha=0.7)
        plt.title(f'{profile_name} - Displacements')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (m)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, len(test_results), i + 1 + len(test_results))
        plt.plot(data['time'], data['actions'])
        plt.title(f'{profile_name} - Control Actions')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Training and testing completed!")