import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from Suspension_Model import QuarterCarModel
from Road_profile import SquareWaveProfile, BumpProfile

# Fix for potential Matplotlib OMP error on some systems
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class QuadraticQLearning:
    """
    Implements Quadratic Q-Learning for Active Suspension Control.
    This version incorporates a SCALED multi-objective reward function
    to ensure balanced learning.
    """
    
    def __init__(self, 
                 state_dim=4,
                 action_dim=1, 
                 disturbance_dim=1,
                 learning_rate=0.001, # Slightly reduced learning rate for stability
                 gamma=0.95,
                 exploration_noise=0.2): # Slightly higher initial exploration
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.disturbance_dim = disturbance_dim
        self.total_dim = state_dim + action_dim + disturbance_dim  # 6D
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_noise = exploration_noise
        
        # Initialize Q-matrix with H∞ block structure
        self._initialize_hinf_q_matrix()
        
        # Optimizer for Q-matrix
        self.optimizer = optim.Adam([self.Q_matrix], lr=learning_rate)
        
        # --- REWARD FUNCTION PARAMETERS (CRITICAL CHANGE) ---
        # Multi-objective weights
        self.weight_comfort = 0.4
        self.weight_energy = 0.3
        self.weight_handling = 0.3
        
        # Normalization factors to scale reward components to a similar range (~[-1, 1])
        # These values prevent one objective from dominating the others.
        self.comfort_scale = 100.0    # Expected max for acceleration² (m²/s⁴)
        self.energy_scale = 20.0      # Expected max for power (W)
        self.handling_scale = 0.001   # Expected max for tire deflection² (m²)
        self.power_consumption_scale = 5.0 # Expected max for power consumed
        
        # Experience replay buffer
        self.memory = deque(maxlen=20000) # Increased memory size
        self.batch_size = 64 # Increased batch size

    def _initialize_hinf_q_matrix(self):
        """
        Initializes the Q-matrix with H∞ control structure.
        Structure: z = [x_s, x_s_dot, x_u, x_u_dot, u, x_g]
        """
        self.Q_matrix = torch.zeros(self.total_dim, self.total_dim, dtype=torch.float32)
        
        # Q_xx: State penalty matrix (4×4)
        Q_xx = torch.diag(torch.tensor([1.0, 0.1, 1.0, 0.1]))
        self.Q_matrix[:4, :4] = Q_xx
        
        # Q_uu: Control penalty (scalar) - NEGATIVE for maximization
        self.Q_matrix[4, 4] = -0.01
        
        # Q_ww: Disturbance penalty (scalar) - POSITIVE for minimization
        self.Q_matrix[5, 5] = 0.1
        
        # Initialize off-diagonal blocks with small random values
        self.Q_matrix[:4, 4] = torch.randn(4) * 0.01
        self.Q_matrix[4, :4] = self.Q_matrix[:4, 4]
        self.Q_matrix[:4, 5] = torch.randn(4) * 0.01
        self.Q_matrix[5, :4] = self.Q_matrix[:4, 5]
        self.Q_matrix[4, 5] = torch.randn(1) * 0.01
        self.Q_matrix[5, 4] = self.Q_matrix[4, 5]
        
        self.Q_matrix.requires_grad_(True)

    def compute_q_value(self, z):
        """
        Computes Q(z) = z^T @ Q_matrix @ z for a batch of z vectors.
        """
        if isinstance(z, np.ndarray):
            z = torch.FloatTensor(z)
        
        # *** BUG FIX ***
        # The original code had `self.Q_matrix @ z`, which caused a shape
        # mismatch for batch processing. The correct order is `z @ self.Q_matrix`.
        # This expression now correctly computes the quadratic form for each
        # vector in the batch.
        q_values = torch.sum(z * (z @ self.Q_matrix), dim=-1)
        return q_values
    
    def get_optimal_action(self, state, disturbance):
        """
        Computes optimal action u* = -inv(Q_uu) * (Q_xu^T * x + Q_uw * w)
        """
        state_t = torch.FloatTensor(state)
        disturbance_t = torch.FloatTensor([disturbance])
        
        Q_xu = self.Q_matrix[:4, 4]
        Q_uu = self.Q_matrix[4, 4]
        Q_uw = self.Q_matrix[4, 5]
        
        if abs(Q_uu) > 1e-6:
            optimal_action = -(Q_xu @ state_t + Q_uw * disturbance_t) / Q_uu
        else:
            optimal_action = torch.tensor(0.0)
            
        return optimal_action.item()
    
    def get_action_with_exploration(self, state, disturbance):
        """Gets action with exploration noise for training"""
        optimal_action = self.get_optimal_action(state, disturbance)
        noise = np.random.normal(0, self.exploration_noise)
        action = optimal_action + noise
        return np.clip(action, -100.0, 100.0)
    
    def compute_scaled_reward(self, state, action, p_regen, x_s_ddot):
        """
        Computes the SCALED multi-objective reward.
        This ensures all reward components are on a similar magnitude.
        """
        # 1. Comfort Cost (Jk): Penalize high body acceleration
        comfort_cost = (x_s_ddot**2) / self.comfort_scale
        
        # 2. Energy Reward (Je): Reward regeneration, penalize consumption
        power_consumed = abs(action * (state[1] - state[3]))
        energy_reward = (p_regen / self.energy_scale) - (power_consumed / self.power_consumption_scale)
        
        # 3. Handling Cost (Jg): Penalize high tire deflection
        tire_deflection = abs(state[2])
        handling_cost = (tire_deflection**2) / self.handling_scale
        
        # 4. Combined, weighted, and scaled reward
        reward = (self.weight_comfort * (-comfort_cost) + 
                  self.weight_energy * energy_reward + 
                  self.weight_handling * (-handling_cost))
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, disturbance, next_disturbance):
        """Stores experience in the replay buffer"""
        self.memory.append((state, action, reward, next_state, disturbance, next_disturbance))
    
    def update_q_matrix(self):
        """Updates Q-matrix using a batch of experiences from memory (vectorized)."""
        if len(self.memory) < self.batch_size:
            return 0.0 # Return 0 loss if not enough samples
        
        # Sample a random batch from memory
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states, actions, rewards, next_states, disturbances, next_disturbances = zip(*batch)
        
        # Convert all batch data to tensors
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        disturbances_t = torch.FloatTensor(disturbances).unsqueeze(1)
        next_disturbances_t = torch.FloatTensor(next_disturbances).unsqueeze(1)
        
        # Form the augmented state vector z for the current batch
        z_current = torch.cat([states_t, actions_t, disturbances_t], dim=1)
        q_current = self.compute_q_value(z_current)
        
        # Compute next Q-value (target) in a vectorized way
        with torch.no_grad():
            # *** PERFORMANCE OPTIMIZATION ***
            # Replaced the slow for-loop with a much faster vectorized calculation.
            Q_xu = self.Q_matrix[:4, 4]
            Q_uu = self.Q_matrix[4, 4]
            Q_uw = self.Q_matrix[4, 5]

            if abs(Q_uu) > 1e-6:
                # Calculate optimal actions for the entire batch at once
                numerator = (next_states_t @ Q_xu) + (Q_uw * next_disturbances_t).squeeze()
                next_actions_t = -(numerator / Q_uu).unsqueeze(1)
            else:
                next_actions_t = torch.zeros_like(actions_t)

            # Form the augmented state vector for the next state batch
            z_next = torch.cat([next_states_t, next_actions_t, next_disturbances_t], dim=1)
            q_next = self.compute_q_value(z_next)
            q_target = rewards_t + self.gamma * q_next
        
        # Compute loss (Mean Squared Bellman Error)
        loss = nn.functional.mse_loss(q_current, q_target)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Enforce H∞ structure constraints after update
        self._enforce_hinf_structure()
        
        return loss.item()

    def _enforce_hinf_structure(self):
        """Enforces symmetry and game-theoretic constraints on Q-matrix"""
        with torch.no_grad():
            self.Q_matrix.data = (self.Q_matrix.data + self.Q_matrix.data.T) / 2
            if self.Q_matrix[4, 4] > -1e-6:
                self.Q_matrix[4, 4] = -1e-6
            if self.Q_matrix[5, 5] < 1e-6:
                self.Q_matrix[5, 5] = 1e-6

    def save_agent(self, filepath):
        """Saves the trained agent's state"""
        save_dict = {
            'Q_matrix': self.Q_matrix.detach().numpy(),
            'hyperparams': {
                'lr': self.learning_rate, 'gamma': self.gamma, 'noise': self.exploration_noise
            },
            'weights': {
                'comfort': self.weight_comfort, 'energy': self.weight_energy, 'handling': self.weight_handling
            }
        }
        np.save(filepath, save_dict)
        print(f"Agent saved to {filepath}")
    
    def load_agent(self, filepath):
        """Loads a trained agent's state"""
        data = np.load(filepath, allow_pickle=True).item()
        self.Q_matrix = torch.FloatTensor(data['Q_matrix'])
        self.Q_matrix.requires_grad_(True)
        self.optimizer = optim.Adam([self.Q_matrix], lr=self.learning_rate)
        print(f"Agent loaded from {filepath}")

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
        self.suspension.reset()
        self.current_step = 0
        self.current_time = 0.0
        return self.suspension.state
    
    def step(self, action):
        road_input = self.road_profile.get_profile(self.current_time)
        next_state, x_s_ddot, p_regen = self.suspension.step(action, road_input)
        self.current_time += self.dt
        self.current_step += 1
        done = self.current_step >= self.max_steps
        next_road_input = self.road_profile.get_profile(self.current_time)
        return next_state, x_s_ddot, p_regen, road_input, next_road_input, done

def train_quadratic_q_learning(episodes=1000, save_path="trained_suspension_agent_v2.npy"):
    """Main training loop with the corrected reward function."""
    
    env = SuspensionEnvironment()
    agent = QuadraticQLearning()
    
    # --- METRICS FOR PLOTTING ---
    metrics = {
        'episode_rewards': [], 'avg_rewards': [], 'td_losses': [],
        'comfort_scores': [], 'energy_scores': [], 'q_matrix_norms': []
    }
    
    print("🚀 Starting Training with SCALED REWARD FUNCTION...")
    print(f"Hyperparams: LR={agent.learning_rate}, Gamma={agent.gamma}, Initial Noise={agent.exploration_noise}")
    print(f"Reward Weights: Comfort={agent.weight_comfort}, Energy={agent.weight_energy}, Handling={agent.weight_handling}")
    print("-" * 60)
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward, episode_comfort, episode_energy, episode_loss = 0, 0, 0, 0
        step_count = 0
        
        road_input = env.road_profile.get_profile(0.0)
        
        while True:
            action = agent.get_action_with_exploration(state, road_input)
            next_state, x_s_ddot, p_regen, current_road, next_road, done = env.step(action)
            
            # *** USE THE SCALED REWARD FUNCTION ***
            reward = agent.compute_scaled_reward(state, action, p_regen, x_s_ddot)
            
            agent.store_experience(state, action, reward, next_state, current_road, next_road)
            
            loss = agent.update_q_matrix()
            
            episode_reward += reward
            episode_comfort += x_s_ddot**2
            episode_energy += p_regen
            episode_loss += loss
            step_count += 1
            
            state = next_state
            road_input = next_road
            
            if done:
                break
        
        # Store metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['comfort_scores'].append(episode_comfort / step_count)
        metrics['energy_scores'].append(episode_energy)
        # Avoid division by zero if episode has no steps
        avg_loss = episode_loss / step_count if step_count > 0 else 0
        metrics['td_losses'].append(avg_loss)
        metrics['q_matrix_norms'].append(torch.norm(agent.Q_matrix).item())
        
        # Calculate moving average reward for better trend visualization
        avg_reward = np.mean(metrics['episode_rewards'][-100:])
        metrics['avg_rewards'].append(avg_reward)
        
        # Decay exploration noise
        agent.exploration_noise = max(0.01, agent.exploration_noise * 0.995)
        
        if episode % 50 == 0 or episode == episodes - 1:
            print(f"Ep {episode:4d}/{episodes} | Avg Reward: {avg_reward:8.3f} | "
                  f"Avg Loss: {metrics['td_losses'][-1]:.4f} | "
                  f"Noise: {agent.exploration_noise:.3f}")
    
    agent.save_agent(save_path)
    print("\n🎉 Training Complete!")
    return agent, metrics

def plot_training_results(metrics):
    """Plots the key metrics from the training process."""
    plt.figure(figsize=(18, 10))
    plt.suptitle("Training Performance with Scaled Rewards", fontsize=16)

    # Plot 1: Episode and Average Rewards
    plt.subplot(2, 3, 1)
    plt.plot(metrics['episode_rewards'], alpha=0.5, label='Episode Reward')
    plt.plot(metrics['avg_rewards'], color='red', linewidth=2, label='100-Ep Avg Reward')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: TD Loss
    plt.subplot(2, 3, 2)
    plt.plot(metrics['td_losses'])
    plt.title('Average TD Loss')
    plt.xlabel('Episode')
    plt.ylabel('MSE Loss')
    plt.grid(True)

    # Plot 3: Comfort Metric
    plt.subplot(2, 3, 3)
    plt.plot(metrics['comfort_scores'])
    plt.title('Comfort Metric (Lower is Better)')
    plt.xlabel('Episode')
    plt.ylabel('Average Acceleration²')
    plt.yscale('log') # Use log scale as it can vary a lot initially
    plt.grid(True)
    
    # Plot 4: Energy Recovery
    plt.subplot(2, 3, 4)
    plt.plot(metrics['energy_scores'])
    plt.title('Energy Recovery')
    plt.xlabel('Episode')
    plt.ylabel('Total Regenerated Power (W)')
    plt.grid(True)
    
    # Plot 5: Q-Matrix Norm
    plt.subplot(2, 3, 5)
    plt.plot(metrics['q_matrix_norms'])
    plt.title('Q-Matrix Evolution (Frobenius Norm)')
    plt.xlabel('Episode')
    plt.ylabel('Norm')
    plt.grid(True)
    
    # Plot 6: Final Q-Matrix
    plt.subplot(2, 3, 6)
    # Need to get the agent from the return of the training function to plot
    # This part will be handled in the main execution block
    plt.title('Final Q-Matrix Structure (placeholder)')
    plt.text(0.5, 0.5, 'Plot after training', ha='center', va='center')


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Train the agent
    agent, training_metrics = train_quadratic_q_learning(episodes=500)
    
    # Plot the results, including the final Q-matrix
    # Get the final Q-matrix from the trained agent
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Performance with Scaled Rewards", fontsize=16)
    
    axes = axes.flatten()

    axes[0].plot(training_metrics['episode_rewards'], alpha=0.5, label='Episode Reward')
    axes[0].plot(training_metrics['avg_rewards'], color='red', linewidth=2, label='100-Ep Avg Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(training_metrics['td_losses'])
    axes[1].set_title('Average TD Loss')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('MSE Loss')
    axes[1].grid(True)

    axes[2].plot(training_metrics['comfort_scores'])
    axes[2].set_title('Comfort Metric (Lower is Better)')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Average Acceleration²')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    axes[3].plot(training_metrics['energy_scores'])
    axes[3].set_title('Energy Recovery')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Total Regenerated Power (W)')
    axes[3].grid(True)
    
    axes[4].plot(training_metrics['q_matrix_norms'])
    axes[4].set_title('Q-Matrix Evolution (Frobenius Norm)')
    axes[4].set_xlabel('Episode')
    axes[4].set_ylabel('Norm')
    axes[4].grid(True)
    
    final_q = agent.Q_matrix.detach().numpy()
    im = axes[5].imshow(final_q, cmap='RdBu_r', vmin=-np.max(np.abs(final_q)), vmax=np.max(np.abs(final_q)))
    axes[5].set_title('Final Q-Matrix Structure')
    labels = ['x_s', 'ẋ_s', 'x_u', 'ẋ_u', 'u', 'x_g']
    axes[5].set_xticks(range(6), labels, rotation=45)
    axes[5].set_yticks(range(6), labels)
    fig.colorbar(im, ax=axes[5])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
    # You can now proceed to test this newly trained agent
    # For example:
    # from reward_analysis import test_trained_agent_comprehensive, plot_comprehensive_results
    # test_results = test_trained_agent_comprehensive("trained_suspension_agent_v2.npy")
    # if test_results:
    #     plot_comprehensive_results(test_results)
