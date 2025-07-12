import numpy as np
import torch
import matplotlib.pyplot as plt
from Suspension_Model import QuarterCarModel
from Road_profile import SquareWaveProfile, BumpProfile

# First, let's fix the OMP error
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class RewardAnalyzer:
    """
    Analyze the reward function components to understand why rewards are so negative
    """
    
    def __init__(self, weight_comfort=0.4, weight_energy=0.3, weight_handling=0.3):
        self.weight_comfort = weight_comfort
        self.weight_energy = weight_energy
        self.weight_handling = weight_handling
        
    def analyze_reward_components(self, states, actions, x_s_ddots, p_regens):
        """
        Break down reward components to understand scaling issues
        """
        comfort_costs = []
        energy_rewards = []
        handling_costs = []
        total_rewards = []
        
        for i in range(len(states)-1):
            state = states[i]
            action = actions[i]
            x_s_ddot = x_s_ddots[i]
            p_regen = p_regens[i]
            
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
            
            comfort_costs.append(comfort_cost)
            energy_rewards.append(energy_reward)
            handling_costs.append(handling_cost)
            total_rewards.append(reward)
        
        return {
            'comfort_costs': np.array(comfort_costs),
            'energy_rewards': np.array(energy_rewards), 
            'handling_costs': np.array(handling_costs),
            'total_rewards': np.array(total_rewards),
            'stats': {
                'avg_comfort_cost': np.mean(comfort_costs),
                'avg_energy_reward': np.mean(energy_rewards),
                'avg_handling_cost': np.mean(handling_costs),
                'avg_total_reward': np.mean(total_rewards),
                'comfort_contribution': np.mean([self.weight_comfort * (-c) for c in comfort_costs]),
                'energy_contribution': np.mean([self.weight_energy * e for e in energy_rewards]),
                'handling_contribution': np.mean([self.weight_handling * (-h) for h in handling_costs])
            }
        }

class ImprovedRewardFunction:
    """
    Improved reward function with better scaling and normalization
    """
    
    def __init__(self, weight_comfort=0.4, weight_energy=0.3, weight_handling=0.3):
        self.weight_comfort = weight_comfort
        self.weight_energy = weight_energy  
        self.weight_handling = weight_handling
        
        # Normalization factors (adjust based on typical ranges)
        self.comfort_scale = 100.0    # Typical acceleration² range: 0-100 m²/s⁴
        self.energy_scale = 10.0      # Typical power range: 0-10 W
        self.handling_scale = 0.001   # Typical deflection² range: 0-0.001 m²
        
    def compute_reward(self, state, action, next_state, p_regen, x_s_ddot):
        """
        Improved reward function with proper scaling
        """
        # Jk: Comfort index (normalize acceleration²)
        comfort_cost = (x_s_ddot**2) / self.comfort_scale
        
        # Je: Energy index (normalize power values)
        power_consumed = abs(action * (state[1] - state[3])) / 1000.0  # Scale down force*velocity
        energy_reward = (p_regen - 0.1 * power_consumed) / self.energy_scale
        
        # Jg: Handling index (normalize tire deflection²)
        tire_deflection = abs(state[2])
        handling_cost = (tire_deflection**2) / self.handling_scale
        
        # Combined reward (all components now roughly in range [-1, 1])
        reward = (self.weight_comfort * (-comfort_cost) + 
                 self.weight_energy * energy_reward + 
                 self.weight_handling * (-handling_cost))
        
        return reward, {
            'comfort_cost': comfort_cost,
            'energy_reward': energy_reward, 
            'handling_cost': handling_cost,
            'comfort_contribution': self.weight_comfort * (-comfort_cost),
            'energy_contribution': self.weight_energy * energy_reward,
            'handling_contribution': self.weight_handling * (-handling_cost)
        }

def test_trained_agent_comprehensive(agent_path="suspension_agent.npy", test_duration=10.0):
    """
    Comprehensive testing of the trained agent
    """
    # Import the QuadraticQLearning class (assuming it's available)
    # For now, let's create a simplified version to load the agent
    
    print("🧪 Loading and Testing Trained Agent")
    print("=" * 50)
    
    # Load the saved agent data
    try:
        agent_data = np.load(agent_path, allow_pickle=True).item()
        Q_matrix = agent_data['Q_matrix']
        weights = agent_data['weights']
        print(f"✅ Agent loaded successfully from {agent_path}")
        print(f"📊 Q-matrix shape: {Q_matrix.shape}")
        print(f"🎯 Weights - Comfort: {weights['comfort']}, Energy: {weights['energy']}, Handling: {weights['handling']}")
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        return None
    
    # Analyze Q-matrix structure
    print(f"\n🔍 Q-Matrix Analysis:")
    print(f"Q_xx (state block) range: [{Q_matrix[:4,:4].min():.6f}, {Q_matrix[:4,:4].max():.6f}]")
    print(f"Q_uu (control penalty): {Q_matrix[4,4]:.6f} (should be negative)")
    print(f"Q_ww (disturbance penalty): {Q_matrix[5,5]:.6f} (should be positive)")
    
    # Create test environment
    suspension = QuarterCarModel(dt=0.001)
    dt = suspension.dt
    max_steps = int(test_duration / dt)
    
    # Define test scenarios using the fixed road profiles from Road_profile.py
    test_scenarios = [
        ("Square Wave (Training)", SquareWaveProfile(period=2.0, amplitude=0.02)),
        ("Large Bump", BumpProfile(start_time=2.0, duration=1.0, height=0.05)),
        ("Small Bump", BumpProfile(start_time=2.0, duration=0.5, height=0.02)),
        ("Rough Road", SquareWaveProfile(period=0.5, amplitude=0.01)),
    ]
    
    results = {}
    reward_analyzer = RewardAnalyzer()
    improved_reward = ImprovedRewardFunction()
    
    for scenario_name, road_profile in test_scenarios:
        print(f"\n🛣️  Testing: {scenario_name}")
        
        # Reset suspension
        suspension.reset()
        state = suspension.state.copy()
        
        # Data storage
        time_history = []
        state_history = []
        action_history = []
        road_history = []
        acceleration_history = []
        power_history = []
        reward_history = []
        improved_reward_history = []
        
        for step in range(max_steps):
            current_time = step * dt
            
            # Get road input
            road_input = road_profile.get_profile(current_time)
            
            # Compute optimal action using loaded Q-matrix
            # u* = -(Q_xu^T * x + Q_uw * w) / Q_uu
            Q_xu = Q_matrix[:4, 4]
            Q_uu = Q_matrix[4, 4] 
            Q_uw = Q_matrix[4, 5]
            
            if abs(Q_uu) > 1e-6:
                action = -(Q_xu @ state + Q_uw * road_input) / Q_uu
            else:
                action = 0.0
            
            # Clip action to actuator limits
            action = np.clip(action, -100.0, 100.0)
            
            # Step suspension
            next_state, x_s_ddot, p_regen = suspension.step(action, road_input)
            
            # Compute original reward
            comfort_cost = x_s_ddot**2
            power_consumed = abs(action * (state[1] - state[3]))
            energy_reward = p_regen - 0.1 * power_consumed
            tire_deflection = abs(state[2])
            handling_cost = tire_deflection**2
            
            original_reward = (weights['comfort'] * (-comfort_cost) + 
                             weights['energy'] * energy_reward + 
                             weights['handling'] * (-handling_cost))
            
            # Compute improved reward
            improved_rew, reward_components = improved_reward.compute_reward(
                state, action, next_state, p_regen, x_s_ddot)
            
            # Store data
            time_history.append(current_time)
            state_history.append(state.copy())
            action_history.append(action)
            road_history.append(road_input)
            acceleration_history.append(x_s_ddot)
            power_history.append(p_regen)
            reward_history.append(original_reward)
            improved_reward_history.append(improved_rew)
            
            # Update state
            state = next_state.copy()
        
        # Convert to numpy arrays
        time_history = np.array(time_history)
        state_history = np.array(state_history)
        action_history = np.array(action_history)
        road_history = np.array(road_history)
        acceleration_history = np.array(acceleration_history)
        power_history = np.array(power_history)
        reward_history = np.array(reward_history)
        improved_reward_history = np.array(improved_reward_history)
        
        # Analyze rewards for this scenario
        reward_analysis = reward_analyzer.analyze_reward_components(
            state_history, action_history, acceleration_history, power_history)
        
        # Store results
        results[scenario_name] = {
            'time': time_history,
            'states': state_history,
            'actions': action_history,
            'road_inputs': road_history,
            'accelerations': acceleration_history,
            'power_recovered': power_history,
            'original_rewards': reward_history,
            'improved_rewards': improved_reward_history,
            'reward_analysis': reward_analysis,
            'performance_metrics': {
                'rms_acceleration': np.sqrt(np.mean(acceleration_history**2)),
                'max_acceleration': np.max(np.abs(acceleration_history)),
                'total_energy_recovered': np.sum(power_history) * dt,
                'max_control_force': np.max(np.abs(action_history)),
                'rms_control_force': np.sqrt(np.mean(action_history**2)),
                'original_reward_total': np.sum(reward_history),
                'improved_reward_total': np.sum(improved_reward_history),
                'original_reward_avg': np.mean(reward_history),
                'improved_reward_avg': np.mean(improved_reward_history)
            }
        }
        
        # Print scenario results
        metrics = results[scenario_name]['performance_metrics']
        analysis = results[scenario_name]['reward_analysis']['stats']
        
        print(f"   📈 RMS Acceleration: {metrics['rms_acceleration']:.4f} m/s²")
        print(f"   ⚡ Total Energy Recovered: {metrics['total_energy_recovered']:.3f} J")
        print(f"   🎮 RMS Control Force: {metrics['rms_control_force']:.2f} N")
        print(f"   💰 Original Avg Reward: {metrics['original_reward_avg']:.2f}")
        print(f"   💎 Improved Avg Reward: {metrics['improved_reward_avg']:.4f}")
        print(f"   📊 Reward Breakdown:")
        print(f"      Comfort: {analysis['comfort_contribution']:.2f}")
        print(f"      Energy:  {analysis['energy_contribution']:.2f}")
        print(f"      Handling: {analysis['handling_contribution']:.2f}")
    
    return results

def plot_comprehensive_results(results):
    """
    Create comprehensive plots of the test results
    """
    n_scenarios = len(results)
    fig = plt.figure(figsize=(20, 15))
    
    # Colors for different scenarios
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (scenario_name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        
        # Plot 1: Displacements
        plt.subplot(3, n_scenarios, i + 1)
        plt.plot(data['time'], data['states'][:, 0], label='x_s (sprung)', color=color)
        plt.plot(data['time'], data['states'][:, 2], label='x_u (unsprung)', color=color, alpha=0.7)
        plt.plot(data['time'], data['road_inputs'], label='Road', color='black', alpha=0.5)
        plt.title(f'{scenario_name}\nDisplacements')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (m)')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Control Actions
        plt.subplot(3, n_scenarios, i + 1 + n_scenarios)
        plt.plot(data['time'], data['actions'], color=color)
        plt.title(f'{scenario_name}\nControl Actions')
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.grid(True)
        
        # Plot 3: Rewards Comparison
        plt.subplot(3, n_scenarios, i + 1 + 2*n_scenarios)
        plt.plot(data['time'], data['original_rewards'], label='Original', color='red', alpha=0.7)
        plt.plot(data['time'], data['improved_rewards'], label='Improved', color='blue', alpha=0.7)
        plt.title(f'{scenario_name}\nReward Comparison')
        plt.xlabel('Time (s)')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Create summary comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    scenarios = list(results.keys())
    
    # Performance metrics comparison
    rms_accel = [results[s]['performance_metrics']['rms_acceleration'] for s in scenarios]
    energy_recovered = [results[s]['performance_metrics']['total_energy_recovered'] for s in scenarios]
    original_rewards = [results[s]['performance_metrics']['original_reward_avg'] for s in scenarios]
    improved_rewards = [results[s]['performance_metrics']['improved_reward_avg'] for s in scenarios]
    
    axes[0,0].bar(range(len(scenarios)), rms_accel)
    axes[0,0].set_title('RMS Acceleration (Lower = Better Comfort)')
    axes[0,0].set_xlabel('Scenario')
    axes[0,0].set_ylabel('RMS Acceleration (m/s²)')
    axes[0,0].set_xticks(range(len(scenarios)))
    axes[0,0].set_xticklabels([s[:10] for s in scenarios], rotation=45)
    
    axes[0,1].bar(range(len(scenarios)), energy_recovered)
    axes[0,1].set_title('Total Energy Recovered')
    axes[0,1].set_xlabel('Scenario')
    axes[0,1].set_ylabel('Energy (J)')
    axes[0,1].set_xticks(range(len(scenarios)))
    axes[0,1].set_xticklabels([s[:10] for s in scenarios], rotation=45)
    
    x = np.arange(len(scenarios))
    width = 0.35
    axes[1,0].bar(x - width/2, original_rewards, width, label='Original', alpha=0.7)
    axes[1,0].bar(x + width/2, improved_rewards, width, label='Improved', alpha=0.7)
    axes[1,0].set_title('Average Reward Comparison')
    axes[1,0].set_xlabel('Scenario')
    axes[1,0].set_ylabel('Average Reward')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([s[:10] for s in scenarios], rotation=45)
    axes[1,0].legend()
    
    # Reward component breakdown for first scenario
    first_scenario = scenarios[0]
    analysis = results[first_scenario]['reward_analysis']['stats']
    components = ['Comfort', 'Energy', 'Handling']
    values = [analysis['comfort_contribution'], analysis['energy_contribution'], analysis['handling_contribution']]
    
    axes[1,1].bar(components, values)
    axes[1,1].set_title(f'Reward Components\n({first_scenario})')
    axes[1,1].set_ylabel('Contribution to Total Reward')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🔍 Investigating Reward Function and Testing Agent")
    print("=" * 60)
    
    # Test the trained agent
    results = test_trained_agent_comprehensive("suspension_agent.npy", test_duration=5.0)
    
    if results is not None:
        # Plot comprehensive results
        plot_comprehensive_results(results)
        
        print("\n📊 REWARD FUNCTION ANALYSIS SUMMARY:")
        print("=" * 50)
        
        # Analyze the reward scaling issues
        first_scenario = list(results.keys())[0]
        analysis = results[first_scenario]['reward_analysis']['stats']
        
        print(f"🔍 Issues with Original Reward Function:")
        print(f"   • Comfort cost scale: {analysis['avg_comfort_cost']:.2e} (acceleration²)")
        print(f"   • Energy reward scale: {analysis['avg_energy_reward']:.2e} (power difference)")
        print(f"   • Handling cost scale: {analysis['avg_handling_cost']:.2e} (deflection²)")
        print(f"   • Total reward scale: {analysis['avg_total_reward']:.2e}")
        
        print(f"\n💡 The negative rewards are caused by:")
        print(f"   1. Acceleration² values are very large (0-100+ m²/s⁴)")
        print(f"   2. Tire deflection² values are small but unscaled")
        print(f"   3. Energy values are much smaller than comfort/handling costs")
        print(f"   4. No normalization between different physical units")
        
        print(f"\n✅ Improved reward function shows better scaling:")
        metrics = results[first_scenario]['performance_metrics']
        print(f"   • Original reward average: {metrics['original_reward_avg']:.2f}")
        print(f"   • Improved reward average: {metrics['improved_reward_avg']:.4f}")
        
        print(f"\n🎯 Agent Performance Analysis:")
        for scenario, data in results.items():
            metrics = data['performance_metrics']
            print(f"   {scenario}:")
            print(f"      RMS Acceleration: {metrics['rms_acceleration']:.4f} m/s²")
            print(f"      Energy Recovered: {metrics['total_energy_recovered']:.3f} J")
            print(f"      Max Control Force: {metrics['max_control_force']:.1f} N")
    
    print("\n🎉 Analysis Complete!")