import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import torch
import os
import glob
import re
from collections import deque

# Import your existing classes
from Suspension_Model import QuarterCarModel
from Road_profile import SquareWaveProfile, BumpProfile, ISO8608Profile

class QuadraticQLearning:
    """Minimal Q-Learning class for loading trained agents."""
    def __init__(self):
        self.Q_matrix = None
    
    def load_agent(self, filepath):
        """Loads a trained agent's state."""
        try:
            data = np.load(filepath, allow_pickle=True).item()
            self.Q_matrix = torch.FloatTensor(data['Q_matrix'])
            print(f"✅ Agent loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading agent: {e}")
            return False
    
    def get_action(self, state, disturbance):
        """Computes optimal action using the loaded Q-matrix."""
        if self.Q_matrix is None:
            return 0.0
            
        Q_xu = self.Q_matrix[:4, 4]
        Q_uu = self.Q_matrix[4, 4]
        Q_uw = self.Q_matrix[4, 5]
        
        if abs(Q_uu) > 1e-6:
            action = -(Q_xu @ torch.FloatTensor(state) + Q_uw * disturbance) / Q_uu
            return np.clip(action.item(), -100.0, 100.0)
        return 0.0

class SuspensionSimulation:
    """Real-time 2D visualization of active suspension system."""
    
    def __init__(self, dt=0.005):  # Slower for visualization
        self.dt = dt
        self.suspension = QuarterCarModel(dt=dt)
        self.agent = QuadraticQLearning()
        
        # Load trained agent if available
        self.load_latest_agent()
        
        # Simulation state
        self.time = 0.0
        self.active_control = True
        self.road_speed = 20.0  # m/s (72 km/h)
        self.road_position = 0.0
        
        # Road profiles
        self.road_profiles = {
            'Square Wave': SquareWaveProfile(period=2.0, amplitude=0.02),
            'Large Bump': BumpProfile(start_time=1.0, duration=1.0, height=0.05),
            'Small Bump': BumpProfile(start_time=1.0, duration=0.5, height=0.02),
            'Rough Road': SquareWaveProfile(period=0.5, amplitude=0.01),
        }
        self.current_road_name = 'Square Wave'
        self.current_road = self.road_profiles[self.current_road_name]
        
        # Data history for plotting
        self.max_history = 500
        self.time_history = deque(maxlen=self.max_history)
        self.body_pos_history = deque(maxlen=self.max_history)
        self.wheel_pos_history = deque(maxlen=self.max_history)
        self.road_history = deque(maxlen=self.max_history)
        self.control_history = deque(maxlen=self.max_history)
        self.acceleration_history = deque(maxlen=self.max_history)
        
        # Vehicle visual parameters
        self.body_width = 1.5
        self.body_height = 0.4
        self.wheel_radius = 0.3
        self.ground_level = -1.0
        
        self.setup_plot()
        
    def load_latest_agent(self):
        """Load the most recent trained agent."""
        agent_files = []
        if os.path.exists("INTERRUPTED_agent.npy"):
            agent_files.append("INTERRUPTED_agent.npy")
        if os.path.exists("trained_suspension_agent_v3.npy"):
            agent_files.append("trained_suspension_agent_v3.npy")
        if os.path.isdir("checkpoints"):
            agent_files.extend(glob.glob("checkpoints/checkpoint_episode_*.npy"))
        
        if agent_files:
            # Try to load the most recent one
            for filepath in agent_files:
                if self.agent.load_agent(filepath):
                    break
        else:
            print("⚠️ No trained agent found. Using passive control only.")
    
    def setup_plot(self):
        """Initialize the matplotlib figure and subplots."""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main vehicle visualization (top)
        self.ax_main = plt.subplot(3, 2, (1, 2))
        self.ax_main.set_xlim(-2, 8)
        self.ax_main.set_ylim(-2, 2)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('Active Suspension System - Real-Time Simulation')
        self.ax_main.grid(True, alpha=0.3)
        
        # Time-series plots
        self.ax_pos = plt.subplot(3, 2, 3)
        self.ax_pos.set_title('Displacements')
        self.ax_pos.set_ylabel('Position (m)')
        self.ax_pos.grid(True)
        
        self.ax_control = plt.subplot(3, 2, 4)
        self.ax_control.set_title('Control Force')
        self.ax_control.set_ylabel('Force (N)')
        self.ax_control.grid(True)
        
        self.ax_accel = plt.subplot(3, 2, 5)
        self.ax_accel.set_title('Body Acceleration')
        self.ax_accel.set_xlabel('Time (s)')
        self.ax_accel.set_ylabel('Acceleration (m/s²)')
        self.ax_accel.grid(True)
        
        self.ax_metrics = plt.subplot(3, 2, 6)
        self.ax_metrics.set_title('Performance Metrics')
        self.ax_metrics.axis('off')
        
        # Initialize visual elements
        self.init_visual_elements()
        self.setup_controls()
        
    def init_visual_elements(self):
        """Initialize the visual elements of the simulation."""
        # Road surface
        road_x = np.linspace(-2, 8, 200)
        road_y = [self.ground_level + self.get_road_height(x) for x in road_x]
        self.road_line, = self.ax_main.plot(road_x, road_y, 'k-', linewidth=3, label='Road')
        
        # Vehicle body (rectangle)
        self.body_rect = plt.Rectangle((0, 0), self.body_width, self.body_height, 
                                     color='blue', alpha=0.7, label='Vehicle Body')
        self.ax_main.add_patch(self.body_rect)
        
        # Wheel (circle)
        self.wheel = plt.Circle((0, 0), self.wheel_radius, color='gray', label='Wheel')
        self.ax_main.add_patch(self.wheel)
        
        # Spring (zigzag line)
        self.spring_line, = self.ax_main.plot([], [], 'g-', linewidth=3, label='Spring')
        
        # Damper (vertical line with symbol)
        self.damper_line, = self.ax_main.plot([], [], 'r-', linewidth=4, label='Damper')
        
        # Control force arrow
        self.control_arrow = self.ax_main.annotate('', xy=(0, 0), xytext=(0, 0),
                                                 arrowprops=dict(arrowstyle='->', 
                                                               color='red', lw=3))
        
        # Time-series lines
        self.body_line, = self.ax_pos.plot([], [], 'b-', label='Body Position')
        self.wheel_line, = self.ax_pos.plot([], [], 'g--', label='Wheel Position')
        self.road_line_plot, = self.ax_pos.plot([], [], 'k:', label='Road Profile')
        
        self.control_line, = self.ax_control.plot([], [], 'r-', label='Control Force')
        self.accel_line, = self.ax_accel.plot([], [], 'm-', label='Body Acceleration')
        
        # Legends
        self.ax_main.legend(loc='upper right')
        self.ax_pos.legend()
        
    def setup_controls(self):
        """Setup interactive controls."""
        # Add buttons for road profile selection
        button_height = 0.04
        button_width = 0.1
        
        # Road profile buttons
        self.road_buttons = []
        for i, road_name in enumerate(self.road_profiles.keys()):
            ax_button = plt.axes([0.02, 0.95 - i * 0.06, button_width, button_height])
            button = Button(ax_button, road_name)
            button.on_clicked(lambda event, name=road_name: self.change_road_profile(name))
            self.road_buttons.append(button)
        
        # Active/Passive control toggle
        ax_toggle = plt.axes([0.85, 0.95, button_width, button_height])
        self.control_button = Button(ax_toggle, 'Active Control: ON')
        self.control_button.on_clicked(self.toggle_control)
        
        # Speed slider
        ax_speed = plt.axes([0.85, 0.85, 0.12, 0.03])
        self.speed_slider = Slider(ax_speed, 'Speed (m/s)', 5.0, 50.0, valinit=self.road_speed)
        self.speed_slider.on_changed(self.update_speed)
        
    def change_road_profile(self, road_name):
        """Change the current road profile."""
        self.current_road_name = road_name
        self.current_road = self.road_profiles[road_name]
        print(f"🛣️ Changed road profile to: {road_name}")
        
    def toggle_control(self, event):
        """Toggle between active and passive control."""
        self.active_control = not self.active_control
        status = "ON" if self.active_control else "OFF"
        self.control_button.label.set_text(f'Active Control: {status}')
        print(f"🎮 Active control: {status}")
        
    def update_speed(self, val):
        """Update the vehicle speed."""
        self.road_speed = val
        
    def get_road_height(self, x_position):
        """Get road height at a given x position."""
        # Convert x position to time based on speed
        road_time = x_position / self.road_speed
        return self.current_road.get_profile(road_time)
        
    def create_spring_points(self, x_bottom, y_bottom, x_top, y_top, coils=6):
        """Create zigzag points for spring visualization."""
        n_points = coils * 4 + 1
        t = np.linspace(0, 1, n_points)
        
        # Linear interpolation for base line
        x_base = x_bottom + t * (x_top - x_bottom)
        y_base = y_bottom + t * (y_top - y_bottom)
        
        # Add zigzag pattern
        zigzag = 0.1 * np.sin(2 * np.pi * coils * t)
        x_spring = x_base + zigzag * (y_top - y_bottom) / np.sqrt((x_top - x_bottom)**2 + (y_top - y_bottom)**2)
        y_spring = y_base - zigzag * (x_top - x_bottom) / np.sqrt((x_top - x_bottom)**2 + (y_top - y_bottom)**2)
        
        return x_spring, y_spring
        
    def update_simulation(self, frame):
        """Update function called by animation."""
        # Update road position
        self.road_position += self.road_speed * self.dt
        current_road_input = self.current_road.get_profile(self.time)
        
        # Get control action
        if self.active_control and self.agent.Q_matrix is not None:
            control_force = self.agent.get_action(self.suspension.state, current_road_input)
        else:
            control_force = 0.0
            
        # Step simulation
        state, acceleration, _ = self.suspension.step(control_force, current_road_input)
        
        # Update data history
        self.time_history.append(self.time)
        self.body_pos_history.append(state[0])
        self.wheel_pos_history.append(state[2])
        self.road_history.append(current_road_input)
        self.control_history.append(control_force)
        self.acceleration_history.append(acceleration)
        
        # Update visual elements
        self.update_vehicle_visualization(state, control_force)
        self.update_road_visualization()
        self.update_time_series_plots()
        self.update_metrics_display()
        
        self.time += self.dt
        
        return (self.body_rect, self.wheel, self.spring_line, self.damper_line, 
                self.road_line, self.body_line, self.wheel_line, self.road_line_plot,
                self.control_line, self.accel_line)
    
    def update_vehicle_visualization(self, state, control_force):
        """Update the main vehicle visualization."""
        vehicle_x = 3.0  # Fixed x position for vehicle
        
        # Vehicle body position
        body_y = self.ground_level + 0.5 + state[0]  # Add offset for visibility
        self.body_rect.set_xy((vehicle_x - self.body_width/2, body_y))
        
        # Wheel position
        wheel_y = self.ground_level + self.wheel_radius + state[2]
        self.wheel.set_center((vehicle_x, wheel_y))
        
        # Spring connection
        spring_x, spring_y = self.create_spring_points(
            vehicle_x - 0.2, wheel_y + self.wheel_radius,
            vehicle_x - 0.2, body_y
        )
        self.spring_line.set_data(spring_x, spring_y)
        
        # Damper connection
        damper_x = [vehicle_x + 0.2, vehicle_x + 0.2]
        damper_y = [wheel_y + self.wheel_radius, body_y]
        self.damper_line.set_data(damper_x, damper_y)
        
        # Control force arrow
        if abs(control_force) > 1.0:
            arrow_scale = min(abs(control_force) / 50.0, 1.0)
            arrow_dy = arrow_scale * 0.5 * np.sign(control_force)
            self.control_arrow.set_position((vehicle_x + 0.5, body_y + self.body_height/2))
            self.control_arrow.xy = (vehicle_x + 0.5, body_y + self.body_height/2 + arrow_dy)
            self.control_arrow.set_visible(True)
        else:
            self.control_arrow.set_visible(False)
    
    def update_road_visualization(self):
        """Update the road surface visualization."""
        road_x = np.linspace(-2, 8, 200)
        road_y = [self.ground_level + self.get_road_height(x) for x in road_x]
        self.road_line.set_data(road_x, road_y)
    
    def update_time_series_plots(self):
        """Update the time-series plots."""
        if len(self.time_history) < 2:
            return
            
        times = list(self.time_history)
        
        # Position plot
        self.body_line.set_data(times, list(self.body_pos_history))
        self.wheel_line.set_data(times, list(self.wheel_pos_history))
        self.road_line_plot.set_data(times, list(self.road_history))
        
        self.ax_pos.set_xlim(max(0, self.time - 10), self.time + 1)
        if self.body_pos_history:
            y_range = max(max(self.body_pos_history), max(self.wheel_pos_history), max(self.road_history))
            y_min = min(min(self.body_pos_history), min(self.wheel_pos_history), min(self.road_history))
            self.ax_pos.set_ylim(y_min - 0.01, y_range + 0.01)
        
        # Control plot
        self.control_line.set_data(times, list(self.control_history))
        self.ax_control.set_xlim(max(0, self.time - 10), self.time + 1)
        if self.control_history:
            control_range = max(abs(max(self.control_history)), abs(min(self.control_history)))
            self.ax_control.set_ylim(-control_range - 5, control_range + 5)
        
        # Acceleration plot
        self.accel_line.set_data(times, list(self.acceleration_history))
        self.ax_accel.set_xlim(max(0, self.time - 10), self.time + 1)
        if self.acceleration_history:
            accel_range = max(abs(max(self.acceleration_history)), abs(min(self.acceleration_history)))
            self.ax_accel.set_ylim(-accel_range - 0.5, accel_range + 0.5)
    
    def update_metrics_display(self):
        """Update the performance metrics display."""
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        
        if len(self.acceleration_history) > 10:
            recent_accel = list(self.acceleration_history)[-50:]  # Last 50 points
            recent_control = list(self.control_history)[-50:]
            
            rms_accel = np.sqrt(np.mean(np.array(recent_accel)**2))
            max_accel = np.max(np.abs(recent_accel))
            rms_control = np.sqrt(np.mean(np.array(recent_control)**2))
            max_control = np.max(np.abs(recent_control))
            
            metrics_text = f"""Performance Metrics (Recent):
RMS Acceleration: {rms_accel:.3f} m/s²
Max Acceleration: {max_accel:.3f} m/s²
RMS Control Force: {rms_control:.1f} N
Max Control Force: {max_control:.1f} N

Current Settings:
Road Profile: {self.current_road_name}
Control Mode: {'Active' if self.active_control else 'Passive'}
Vehicle Speed: {self.road_speed:.1f} m/s
Time: {self.time:.1f} s"""
            
            self.ax_metrics.text(0.05, 0.95, metrics_text, transform=self.ax_metrics.transAxes,
                               fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def start_simulation(self):
        """Start the real-time simulation."""
        print("🚀 Starting Active Suspension Real-Time Simulation")
        print("Use the buttons to change road profiles and toggle control modes!")
        
        # Reset simulation
        self.suspension.reset()
        self.time = 0.0
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_simulation, interval=50,  # 20 FPS
            blit=False, cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    # Create and start the simulation
    sim = SuspensionSimulation(dt=0.001)  # 10ms timestep for smooth animation
    sim.start_simulation()