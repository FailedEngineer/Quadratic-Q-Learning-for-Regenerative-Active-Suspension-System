import numpy as np
import torch
import matplotlib.pyplot as plt
from Suspension_Model import QuarterCarModel
from Road_profile import SquareWaveProfile, BumpProfile
import os
import glob
import re

# Fix for potential Matplotlib OMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Note: The QuadraticQLearning class is needed to load the agent
# It's assumed to be available via import or defined in the same scope.
# For simplicity, we can define a minimal version here if running standalone.
class QuadraticQLearning:
    """Minimal class definition for loading the agent's Q-matrix."""
    def __init__(self):
        self.Q_matrix = None
    
    def load_agent(self, filepath):
        """Loads a trained agent's state."""
        try:
            data = np.load(filepath, allow_pickle=True).item()
            self.Q_matrix = torch.FloatTensor(data['Q_matrix'])
            print(f"✅ Agent Q-matrix loaded successfully from {filepath}")
            return data
        except Exception as e:
            print(f"❌ Error loading agent: {e}")
            return None

def find_latest_agent_file(checkpoint_dir="checkpoints", final_agent_name="trained_suspension_agent_v3.npy"):
    """
    Finds the latest trained agent file.
    Priority: Interrupted > Final > Latest Checkpoint
    """
    interrupted_agent = "INTERRUPTED_agent.npy"
    if os.path.exists(interrupted_agent):
        print(f"Found interrupted session file: {interrupted_agent}")
        return interrupted_agent

    if os.path.exists(final_agent_name):
        print(f"Found final trained agent: {final_agent_name}")
        return final_agent_name

    if not os.path.isdir(checkpoint_dir):
        return None

    list_of_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_episode_*.npy'))
    if not list_of_files:
        return None

    # Find the file with the highest episode number in its name
    latest_file = max(list_of_files, key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(1)))
    return latest_file

def test_trained_agent_comprehensive(agent_path, test_duration=10.0):
    """
    Comprehensive testing of the trained agent.
    """
    print("🧪 Loading and Testing Trained Agent")
    print("=" * 50)
    
    # Load the saved agent data
    agent = QuadraticQLearning()
    agent_data = agent.load_agent(agent_path)
    if agent_data is None:
        return None
        
    Q_matrix = agent.Q_matrix.numpy() # Convert to numpy for analysis
    weights = agent_data.get('weights', {'comfort': 0.4, 'energy': 0.3, 'handling': 0.3}) # Default weights if not in file
    
    print(f"📊 Q-matrix shape: {Q_matrix.shape}")
    print(f"🎯 Weights - Comfort: {weights['comfort']}, Energy: {weights['energy']}, Handling: {weights['handling']}")
    
    # Create test environment
    suspension = QuarterCarModel(dt=0.001)
    dt = suspension.dt
    max_steps = int(test_duration / dt)
    
    # Define test scenarios
    test_scenarios = [
        ("Square Wave (Training)", SquareWaveProfile(period=2.0, amplitude=0.02)),
        ("Large Bump", BumpProfile(start_time=2.0, duration=1.0, height=0.05)),
        ("Small Bump", BumpProfile(start_time=2.0, duration=0.5, height=0.02)),
        ("Rough Road", SquareWaveProfile(period=0.5, amplitude=0.01)),
    ]
    
    results = {}
    
    for scenario_name, road_profile in test_scenarios:
        print(f"\n🛣️  Testing: {scenario_name}")
        
        suspension.reset()
        state = suspension.state.copy()
        
        time_history, state_history, action_history, road_history, acceleration_history = [], [], [], [], []
        
        for step in range(max_steps):
            current_time = step * dt
            road_input = road_profile.get_profile(current_time)
            
            # Compute optimal action using loaded Q-matrix
            Q_xu = Q_matrix[:4, 4]
            Q_uu = Q_matrix[4, 4] 
            Q_uw = Q_matrix[4, 5]
            
            if abs(Q_uu) > 1e-6:
                action = -(Q_xu @ state + Q_uw * road_input) / Q_uu
            else:
                action = 0.0
            
            action = np.clip(action, -100.0, 100.0)
            
            next_state, x_s_ddot, p_regen = suspension.step(action, road_input)
            
            time_history.append(current_time)
            state_history.append(state.copy())
            action_history.append(action)
            road_history.append(road_input)
            acceleration_history.append(x_s_ddot)
            
            state = next_state.copy()
        
        # Convert to numpy arrays
        time_history = np.array(time_history)
        state_history = np.array(state_history)
        action_history = np.array(action_history)
        road_history = np.array(road_history)
        acceleration_history = np.array(acceleration_history)
        
        results[scenario_name] = {
            'time': time_history,
            'states': state_history,
            'actions': action_history,
            'road_inputs': road_history,
            'accelerations': acceleration_history,
            'performance_metrics': {
                'rms_acceleration': np.sqrt(np.mean(acceleration_history**2)),
                'max_acceleration': np.max(np.abs(acceleration_history)),
                'max_control_force': np.max(np.abs(action_history)),
                'rms_control_force': np.sqrt(np.mean(action_history**2)),
            }
        }
        
        metrics = results[scenario_name]['performance_metrics']
        print(f"   📈 RMS Acceleration: {metrics['rms_acceleration']:.4f} m/s²")
        print(f"   🎮 RMS Control Force: {metrics['rms_control_force']:.2f} N")
    
    return results

def plot_comprehensive_results(results):
    """Create comprehensive plots of the test results."""
    n_scenarios = len(results)
    fig, axes = plt.subplots(2, n_scenarios, figsize=(20, 10), sharey='row')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (scenario_name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        
        # Plot 1: Displacements
        ax1 = axes[0, i]
        ax1.plot(data['time'], data['states'][:, 0] * 100, label='x_s (sprung)', color=color)
        ax1.plot(data['time'], data['states'][:, 2] * 100, label='x_u (unsprung)', color=color, alpha=0.7, linestyle='--')
        ax1.plot(data['time'], data['road_inputs'] * 100, label='Road', color='black', alpha=0.5, linestyle=':')
        ax1.set_title(f'{scenario_name}\nDisplacements')
        ax1.set_xlabel('Time (s)')
        if i == 0: ax1.set_ylabel('Displacement (cm)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Control Actions
        ax2 = axes[1, i]
        ax2.plot(data['time'], data['actions'], color=color)
        ax2.set_title(f'{scenario_name}\nControl Actions')
        ax2.set_xlabel('Time (s)')
        if i == 0: ax2.set_ylabel('Force (N)')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🔍 Evaluating Latest Trained Agent")
    print("=" * 60)
    
    # --- NEW: Automatically find the latest agent to test ---
    latest_agent_path = find_latest_agent_file()
    
    if latest_agent_path is None:
        print("❌ No trained agent found in the current directory or in './checkpoints/'.")
        print("   Please run Qlearning_Training.py first.")
    else:
        # Test the found agent
        test_results = test_trained_agent_comprehensive(agent_path=latest_agent_path, test_duration=5.0)
    
        if test_results is not None:
            # Plot comprehensive results
            plot_comprehensive_results(test_results)
