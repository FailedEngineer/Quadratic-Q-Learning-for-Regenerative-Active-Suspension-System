import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import os
import sys
import glob
import re
from Suspension_Model import QuarterCarModel
from Road_profile import SquareWaveProfile, BumpProfile

# Fix for potential Matplotlib OMP error on some systems
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
                 learning_rate=0.0001,
                 gamma=0.95,
                 exploration_noise=0.2):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.disturbance_dim = disturbance_dim
        self.total_dim = state_dim + action_dim + disturbance_dim  # 6D
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_noise = exploration_noise
        
        self._initialize_hinf_q_matrix()
        self.optimizer = optim.Adam([self.Q_matrix], lr=learning_rate)
        
        # --- REWARD FUNCTION PARAMETERS (RE-BALANCED) ---
        self.weight_comfort = 0.4
        self.weight_energy = 0.2
        self.weight_handling = 0.4
        
        # --- NEW SCALING FACTORS ---
        # These factors normalize the costs. Adjust them based on typical
        # values observed during a passive run to make them comparable.
        self.comfort_scale = 10.0    # Expected RMS of body acceleration
        self.handling_scale = 0.05  # Expected RMS of tire deflection
        self.power_consumption_scale = 100.0 # Expected RMS of power consumed
        self.energy_regen_scale = 50.0 # Expected RMS of power regenerated

        self.memory = deque(maxlen=20000)
        self.batch_size = 64

    def _initialize_hinf_q_matrix(self):
        """Initializes the Q-matrix with H∞ control structure."""
        self.Q_matrix = torch.zeros(self.total_dim, self.total_dim, dtype=torch.float32)
        Q_xx = torch.diag(torch.tensor([1.0, 0.1, 1.0, 0.1]))
        self.Q_matrix[:4, :4] = Q_xx
        self.Q_matrix[4, 4] = -0.01
        self.Q_matrix[5, 5] = 0.1
        self.Q_matrix[:4, 4] = torch.randn(4) * 0.01
        self.Q_matrix[4, :4] = self.Q_matrix[:4, 4]
        self.Q_matrix[:4, 5] = torch.randn(4) * 0.01
        self.Q_matrix[5, :4] = self.Q_matrix[:4, 5]
        self.Q_matrix[4, 5] = torch.randn(1) * 0.01
        self.Q_matrix[5, 4] = self.Q_matrix[4, 5]
        self.Q_matrix.requires_grad_(True)

    def compute_q_value(self, z):
        """Computes Q(z) = z^T @ Q_matrix @ z for a batch of z vectors."""
        if isinstance(z, np.ndarray):
            z = torch.FloatTensor(z)
        q_values = torch.sum(z * (z @ self.Q_matrix), dim=-1)
        return q_values
    
    def get_optimal_action(self, state, disturbance):
        """Computes optimal action u*."""
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
        """Gets action with exploration noise for training."""
        optimal_action = self.get_optimal_action(state, disturbance)
        noise = np.random.normal(0, self.exploration_noise)
        action = optimal_action + noise
        return np.clip(action, -100.0, 100.0)
    
    def compute_scaled_reward(self, state, action, x_s_ddot, p_regen, x_g):
        """
        Computes the SCALED multi-objective reward based on the project proposal.
        J_k: Passenger Comfort, J_g: Road Handling, J_e: Energy
        """
        # 1. Comfort Cost (J_k): Penalize body acceleration (m/s^2)
        comfort_cost = (x_s_ddot**2) / self.comfort_scale**2

        # 2. Handling Cost (J_g): Penalize tire deflection from the road (m)
        #    This is the difference between the wheel's position and the road's position.
        tire_deflection = state[2] - x_g  # x_u - x_g
        handling_cost = (tire_deflection**2) / self.handling_scale**2
        
        # 3. Net Energy Cost (J_e): Penalize actuator power usage and reward regeneration
        relative_velocity = state[1] - state[3]  # x_s_dot - x_u_dot
        power_consumed = (action * relative_velocity)**2 / self.power_consumption_scale**2
        power_regenerated = p_regen / self.energy_regen_scale
        
        # The goal is to MINIMIZE cost, so rewards are negative costs.
        # We maximize the reward: R = -w_c*J_k - w_h*J_g + w_e*J_e_net
        reward = (self.weight_comfort * (-comfort_cost) +
                  self.weight_handling * (-handling_cost) +
                  self.weight_energy * (power_regenerated - power_consumed))

        return reward
    
    def store_experience(self, state, action, reward, next_state, disturbance, next_disturbance):
        """Stores experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, disturbance, next_disturbance))
    
    def update_q_matrix(self):
        """Updates Q-matrix using a batch of experiences from memory (vectorized)."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        states, actions, rewards, next_states, disturbances, next_disturbances = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        disturbances_t = torch.FloatTensor(disturbances).unsqueeze(1)
        next_disturbances_t = torch.FloatTensor(next_disturbances).unsqueeze(1)
        
        z_current = torch.cat([states_t, actions_t, disturbances_t], dim=1)
        q_current = self.compute_q_value(z_current)
        
        with torch.no_grad():
            Q_xu = self.Q_matrix[:4, 4]
            Q_uu = self.Q_matrix[4, 4]
            Q_uw = self.Q_matrix[4, 5]
            if abs(Q_uu) > 1e-6:
                numerator = (next_states_t @ Q_xu) + (Q_uw * next_disturbances_t).squeeze()
                next_actions_t = -(numerator / Q_uu).unsqueeze(1)
            else:
                next_actions_t = torch.zeros_like(actions_t)
            z_next = torch.cat([next_states_t, next_actions_t, next_disturbances_t], dim=1)
            q_next = self.compute_q_value(z_next)
            q_target = rewards_t + self.gamma * q_next
        
        loss = nn.functional.mse_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q_matrix, 1.0)
        self.optimizer.step()
        self._enforce_hinf_structure()
        return loss.item()

    def _enforce_hinf_structure(self):
        """Enforces symmetry and game-theoretic constraints on Q-matrix."""
        with torch.no_grad():
            self.Q_matrix.data = (self.Q_matrix.data + self.Q_matrix.data.T) / 2
            if self.Q_matrix[4, 4] > -1e-6:
                self.Q_matrix[4, 4] = -1e-6
            if self.Q_matrix[5, 5] < 1e-6:
                self.Q_matrix[5, 5] = 1e-6

    def save_agent(self, filepath, episode_num):
        """Saves the trained agent's state and current episode number."""
        save_dict = {
            'episode': episode_num,
            'Q_matrix': self.Q_matrix.detach().numpy(),
            'hyperparams': {'lr': self.learning_rate, 'gamma': self.gamma, 'noise': self.exploration_noise},
            'weights': {'comfort': self.weight_comfort, 'energy': self.weight_energy, 'handling': self.weight_handling}
        }
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        np.save(filepath, save_dict)
        print(f"Agent state for episode {episode_num} saved to {filepath}")
    
    def load_agent(self, filepath):
        """Loads a trained agent's state and returns the episode number."""
        data = np.load(filepath, allow_pickle=True).item()
        self.Q_matrix = torch.FloatTensor(data['Q_matrix'])
        self.Q_matrix.requires_grad_(True)
        self.optimizer = optim.Adam([self.Q_matrix], lr=self.learning_rate)
        if 'hyperparams' in data:
            self.exploration_noise = data['hyperparams'].get('noise', self.exploration_noise)
        
        episode_num = data.get('episode', 1000)
        print(f"Agent state loaded from {filepath} (trained up to episode {episode_num}).")
        return episode_num

class SuspensionEnvironment:
    """Environment wrapper for the suspension system."""
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

def plot_training_results(metrics, agent, save_path=None):
    """
    Plots the key metrics from the training process.
    If save_path is provided, saves the plot instead of showing it.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Training Performance with Scaled Rewards", fontsize=16)
    axes = axes.flatten()
    
    axes[0].plot(metrics['episode_rewards'], alpha=0.5, label='Episode Reward')
    axes[0].plot(metrics['avg_rewards'], color='red', linewidth=2, label='100-Ep Avg Reward')
    axes[0].set_title('Episode Rewards'); axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Total Reward'); axes[0].legend(); axes[0].grid(True)
    
    axes[1].plot(metrics['td_losses']); axes[1].set_title('Average TD Loss'); axes[1].set_xlabel('Episode'); axes[1].set_ylabel('MSE Loss'); axes[1].grid(True)
    
    axes[2].plot(metrics['comfort_scores']); axes[2].set_title('Comfort Metric (Lower is Better)'); axes[2].set_xlabel('Episode'); axes[2].set_ylabel('Average Acceleration²'); axes[2].set_yscale('log'); axes[2].grid(True)
    
    axes[3].plot(metrics['energy_scores']); axes[3].set_title('Energy Recovery'); axes[3].set_xlabel('Episode'); axes[3].set_ylabel('Total Regenerated Power (W)'); axes[3].grid(True)
    
    axes[4].plot(metrics['q_matrix_norms']); axes[4].set_title('Q-Matrix Evolution (Frobenius Norm)'); axes[4].set_xlabel('Episode'); axes[4].set_ylabel('Norm'); axes[4].grid(True)
    
    final_q = agent.Q_matrix.detach().numpy()
    im = axes[5].imshow(final_q, cmap='RdBu_r', vmin=-np.max(np.abs(final_q)), vmax=np.max(np.abs(final_q)))
    axes[5].set_title('Final Q-Matrix Structure')
    labels = ['x_s', 'ẋ_s', 'x_u', 'ẋ_u', 'u', 'x_g']
    axes[5].set_xticks(range(6), labels, rotation=45); axes[5].set_yticks(range(6), labels); fig.colorbar(im, ax=axes[5])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def find_and_load_latest_agent(agent, checkpoint_dir="checkpoints", final_agent_name="trained_suspension_agent_v3.npy"):
    """
    Finds the agent with the highest episode number, loads it, and returns the next episode to run.
    This is more robust than checking modification times.
    """
    candidate_files = {}

    # Helper function to safely read the episode number from a saved agent file
    def get_episode_from_file(filepath):
        try:
            data = np.load(filepath, allow_pickle=True).item()
            return data.get('episode', 0)
        except Exception:
            return 0

    # 1. Gather all possible agent files
    all_files_to_check = []
    if os.path.exists("INTERRUPTED_agent.npy"):
        all_files_to_check.append("INTERRUPTED_agent.npy")
    if os.path.exists(final_agent_name):
        all_files_to_check.append(final_agent_name)
    if os.path.isdir(checkpoint_dir):
        all_files_to_check.extend(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_episode_*.npy')))

    if not all_files_to_check:
        print("No saved agent found. Starting training from scratch.")
        return 0

    # 2. For each file, read the episode number stored inside it
    for f in all_files_to_check:
        candidate_files[f] = get_episode_from_file(f)

    # 3. Find the file path corresponding to the highest episode number
    if not candidate_files:
        print("Could not read episode number from any saved agent. Starting from scratch.")
        return 0
        
    latest_file = max(candidate_files, key=candidate_files.get)
    
    # 4. Load the agent and return the next episode number to start from
    print(f"Resuming training from the most advanced state found: {latest_file}")
    last_episode = agent.load_agent(latest_file)
    
    return last_episode

def train_quadratic_q_learning(episodes=1000, save_path="trained_suspension_agent_v3.npy", checkpoint_interval=100, plot_interval=100):
    """Main training loop with auto-resume, checkpointing, and non-blocking plotting."""
    
    agent = QuadraticQLearning()
    
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # *** NEW: Use the robust resume logic ***
    start_episode = find_and_load_latest_agent(agent)
    
    if start_episode >= episodes:
        print(f"Training already completed up to episode {start_episode}. Target is {episodes}. Exiting.")
        return agent, None

    env = SuspensionEnvironment()
    metrics = {'episode_rewards': [], 'avg_rewards': [], 'td_losses': [], 'comfort_scores': [], 'energy_scores': [], 'q_matrix_norms': []}
    
    print("🚀 Starting Training with SCALED REWARD FUNCTION...")
    print(f"Hyperparams: LR={agent.learning_rate}, Gamma={agent.gamma}, Noise={agent.exploration_noise}")
    print("Press Ctrl+C to interrupt training and save the current agent state.")
    print("-" * 60)
    
    try:
        for episode in range(start_episode, episodes):
            state = env.reset()
            episode_reward, episode_comfort, episode_energy, episode_loss = 0, 0, 0, 0
            step_count = 0
            road_input = env.road_profile.get_profile(0.0)
            
            while True:
                action = agent.get_action_with_exploration(state, road_input)
                next_state, x_s_ddot, p_regen, current_road, next_road, done = env.step(action)
                reward = agent.compute_scaled_reward(state, action,x_s_ddot, p_regen,current_road)
                agent.store_experience(state, action, reward, next_state, current_road, next_road)
                loss = agent.update_q_matrix()
                
                episode_reward += reward; episode_comfort += x_s_ddot**2; episode_energy += p_regen; episode_loss += loss
                step_count += 1; state = next_state; road_input = next_road
                if done: break
            
            metrics['episode_rewards'].append(episode_reward); metrics['comfort_scores'].append(episode_comfort / step_count)
            metrics['energy_scores'].append(episode_energy); avg_loss = episode_loss / step_count if step_count > 0 else 0
            metrics['td_losses'].append(avg_loss); metrics['q_matrix_norms'].append(torch.norm(agent.Q_matrix).item())
            
            avg_reward = np.mean(metrics['episode_rewards'][-100:])
            metrics['avg_rewards'].append(avg_reward)
            agent.exploration_noise = max(0.01, agent.exploration_noise * 0.995)
            
            current_episode_num = episode + 1
            #if current_episode_num % 50 == 0 or episode == episodes - 1:
            print(f"Ep {current_episode_num:4d}/{episodes} | Avg Reward: {avg_reward:8.3f} | Avg Loss: {metrics['td_losses'][-1]:.4f} | Noise: {agent.exploration_noise:.3f}")

            if current_episode_num % checkpoint_interval == 0:
                checkpoint_path = os.path.join("checkpoints", f"checkpoint_episode_{current_episode_num}.npy")
                agent.save_agent(checkpoint_path, current_episode_num)

            if current_episode_num % plot_interval == 0:
                plot_path = os.path.join(plots_dir, f"progress_episode_{current_episode_num}.png")
                plot_training_results(metrics, agent, save_path=plot_path)

    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user.")
        interrupted_save_path = "INTERRUPTED_agent.npy"
        agent.save_agent(interrupted_save_path, episode)
        print("Gracefully exiting.")
        sys.exit(0)

    agent.save_agent(save_path, episodes)
    print("\n🎉 Training Complete!")
    return agent, metrics

# --- Main Execution ---
if __name__ == "__main__":
    agent, training_metrics = train_quadratic_q_learning(episodes=15000)
    
    if training_metrics:
        plot_training_results(training_metrics, agent, save_path="final_training_plot.png")
