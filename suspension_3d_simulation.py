# Fix OpenMP conflict BEFORE any other imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, Slider
import torch
import glob
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

class SuspensionSimulation3D:
    """3D Real-time visualization of active suspension system."""
    
    def __init__(self, dt=0.01):
        self.dt = dt
        self.suspension = QuarterCarModel(dt=dt)
        self.agent = QuadraticQLearning()
        
        # Load trained agent if available
        self.load_latest_agent()
        
        # Simulation state
        self.time = 0.0
        self.active_control = True
        self.road_speed = 20.0  # m/s
        
        # Road profiles
        self.road_profiles = {
            'Square Wave': SquareWaveProfile(period=2.0, amplitude=0.02),
            'Large Bump': BumpProfile(start_time=1.0, duration=1.0, height=0.05),
            'Small Bump': BumpProfile(start_time=1.0, duration=0.5, height=0.02),
            'Rough Road': SquareWaveProfile(period=0.5, amplitude=0.01),
        }
        self.current_road_name = 'Square Wave'
        self.current_road = self.road_profiles[self.current_road_name]
        
        # Vehicle parameters for 3D visualization
        self.vehicle_length = 2.0
        self.vehicle_width = 1.0
        self.vehicle_height = 0.3
        self.wheel_radius = 0.25
        
        # Road visualization parameters
        self.road_length = 20.0
        self.road_width = 2.0
        self.road_points = 100
        
        # Data history
        self.max_history = 200
        self.time_history = deque(maxlen=self.max_history)
        self.body_pos_history = deque(maxlen=self.max_history)
        self.wheel_pos_history = deque(maxlen=self.max_history)
        self.road_history = deque(maxlen=self.max_history)
        self.control_history = deque(maxlen=self.max_history)
        
        # Initialize 3D elements
        self.ax_3d = None
        self.road_surface = None
        self.vehicle_body = None
        self.wheels = []
        
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
            for filepath in agent_files:
                if self.agent.load_agent(filepath):
                    break
        else:
            print("⚠️ No trained agent found. Using passive control only.")
    
    def change_road_profile(self, road_name):
        """Change the current road profile."""
        if road_name in self.road_profiles:
            self.current_road_name = road_name
            self.current_road = self.road_profiles[road_name]
            # Only update road surface if plot is already initialized
            if hasattr(self, 'ax_3d') and self.ax_3d is not None:
                self.update_road_surface()
            print(f"🛣️ Changed road profile to: {road_name}")
        else:
            print(f"⚠️ Unknown road profile: {road_name}")
            print(f"Available profiles: {list(self.road_profiles.keys())}")
    
    def setup_plot(self):
        """Initialize the 3D plot."""
        plt.style.use('default')  # Ensure consistent styling
        
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('3D Active Suspension System Simulation', fontsize=16, fontweight='bold')
        
        # Main 3D plot
        self.ax_3d = self.fig.add_subplot(2, 2, (1, 2), projection='3d')
        self.ax_3d.set_title('3D Vehicle Dynamics')
        
        # Performance plots
        self.ax_pos = self.fig.add_subplot(2, 2, 3)
        self.ax_pos.set_title('Vehicle Response')
        self.ax_pos.set_ylabel('Position (m)')
        self.ax_pos.grid(True)
        
        self.ax_control = self.fig.add_subplot(2, 2, 4)
        self.ax_control.set_title('Control & Metrics')
        self.ax_control.set_ylabel('Force (N)')
        self.ax_control.set_xlabel('Time (s)')
        self.ax_control.grid(True)
        
        self.init_3d_elements()
        self.setup_controls()
        
    def init_3d_elements(self):
        """Initialize 3D visual elements."""
        # Create initial road surface
        self.update_road_surface()
        
        # Initialize empty plots for time series
        self.body_line, = self.ax_pos.plot([], [], 'b-', label='Body Position', linewidth=2)
        self.wheel_line, = self.ax_pos.plot([], [], 'g--', label='Wheel Position', linewidth=2)
        self.road_line_plot, = self.ax_pos.plot([], [], 'k:', label='Road Profile', linewidth=1)
        
        self.control_line, = self.ax_control.plot([], [], 'r-', label='Control Force', linewidth=2)
        
        self.ax_pos.legend()
        self.ax_control.legend()
        
        # Set 3D plot properties
        self.ax_3d.set_xlim(0, self.road_length)
        self.ax_3d.set_ylim(-self.road_width/2, self.road_width/2)
        self.ax_3d.set_zlim(-0.2, 0.5)
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
    def setup_controls(self):
        """Setup interactive controls."""
        button_height = 0.03
        button_width = 0.08
        
        # Road profile buttons
        self.road_buttons = []
        for i, road_name in enumerate(self.road_profiles.keys()):
            ax_button = plt.axes([0.02, 0.95 - i * 0.04, button_width, button_height])
            button = Button(ax_button, road_name)  # Removed fontsize parameter
            button.on_clicked(lambda event, name=road_name: self.change_road_profile(name))
            self.road_buttons.append(button)
        
        # Control toggle
        ax_toggle = plt.axes([0.02, 0.75, button_width, button_height])
        self.control_button = Button(ax_toggle, 'Active: ON')  # Removed fontsize parameter
        self.control_button.on_clicked(self.toggle_control)
        
        # Speed slider
        ax_speed = plt.axes([0.02, 0.68, 0.15, 0.02])
        self.speed_slider = Slider(ax_speed, 'Speed', 5.0, 50.0, valinit=self.road_speed, valfmt='%.1f m/s')
        self.speed_slider.on_changed(self.update_speed)
    
    def toggle_control(self, event):
        """Toggle between active and passive control."""
        self.active_control = not self.active_control
        status = "ON" if self.active_control else "OFF"
        self.control_button.label.set_text(f'Active: {status}')
        print(f"🎮 Active control: {status}")
        
    def update_speed(self, val):
        """Update vehicle speed."""
        self.road_speed = val
        
    def update_road_surface(self):
        """Update the 3D road surface."""
        # Check if 3D plot is initialized
        if not hasattr(self, 'ax_3d') or self.ax_3d is None:
            print("⚠️ 3D plot not initialized yet, skipping road surface update")
            return
            
        # Clear previous road surface
        if hasattr(self, 'road_surface') and self.road_surface is not None:
            try:
                self.road_surface.remove()
            except:
                pass  # Handle case where surface is already removed
            
        # Create road grid
        x_road = np.linspace(0, self.road_length, self.road_points)
        y_road = np.linspace(-self.road_width/2, self.road_width/2, 20)
        X_road, Y_road = np.meshgrid(x_road, y_road)
        
        # Calculate road heights
        Z_road = np.zeros_like(X_road)
        for i, x in enumerate(x_road):
            road_time = x / self.road_speed
            road_height = self.current_road.get_profile(road_time)
            Z_road[:, i] = road_height
            
        # Plot road surface
        try:
            self.road_surface = self.ax_3d.plot_surface(
                X_road, Y_road, Z_road, alpha=0.6, cmap='terrain', linewidth=0
            )
        except Exception as e:
            print(f"⚠️ Failed to update road surface: {e}")
            self.road_surface = None
        
    def create_vehicle_3d(self, x_pos, body_z, wheel_z):
        """Create or update 3D vehicle representation."""
        # Check if 3D plot is initialized
        if not hasattr(self, 'ax_3d') or self.ax_3d is None:
            return
            
        # Clear previous vehicle elements
        if self.vehicle_body is not None:
            try:
                self.vehicle_body.remove()
            except:
                pass
        for wheel in self.wheels:
            try:
                wheel.remove()
            except:
                pass
        self.wheels.clear()
        
        # Vehicle body (simplified as a rectangular prism)
        body_x = [x_pos - self.vehicle_length/2, x_pos + self.vehicle_length/2]
        body_y = [-self.vehicle_width/2, self.vehicle_width/2]
        body_z_corners = [body_z, body_z + self.vehicle_height]
        
        # Create body wireframe
        body_vertices = []
        for x in body_x:
            for y in body_y:
                for z in body_z_corners:
                    body_vertices.append([x, y, z])
        
        body_vertices = np.array(body_vertices)
        
        # Draw body edges
        edges = [
            [0, 1], [2, 3], [4, 5], [6, 7],  # bottom and top rectangles
            [0, 2], [1, 3], [4, 6], [5, 7],  # vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # connecting edges
        ]
        
        try:
            for edge in edges:
                points = body_vertices[edge]
                self.ax_3d.plot3D(*points.T, 'b-', linewidth=2, alpha=0.8)
            
            # Wheels (as circles)
            wheel_positions = [
                (x_pos - self.vehicle_length/3, -self.vehicle_width/3),
                (x_pos - self.vehicle_length/3, self.vehicle_width/3),
                (x_pos + self.vehicle_length/3, -self.vehicle_width/3),
                (x_pos + self.vehicle_length/3, self.vehicle_width/3)
            ]
            
            for wheel_x, wheel_y in wheel_positions:
                # Create wheel as a circle
                theta = np.linspace(0, 2*np.pi, 20)
                wheel_y_circle = wheel_y + self.wheel_radius * np.cos(theta) * 0.3  # Flattened
                wheel_z_circle = wheel_z + self.wheel_radius * np.sin(theta)
                wheel_x_circle = np.full_like(theta, wheel_x)
                
                wheel_plot = self.ax_3d.plot(wheel_x_circle, wheel_y_circle, wheel_z_circle, 'k-', linewidth=3)
                self.wheels.extend(wheel_plot)
            
            # Spring/damper connections (simplified as lines)
            for wheel_x, wheel_y in wheel_positions:
                # Spring line
                spring_line = self.ax_3d.plot([wheel_x, wheel_x], [wheel_y, wheel_y], 
                                            [wheel_z + self.wheel_radius, body_z], 
                                            'g-', linewidth=2, alpha=0.7)
                self.wheels.extend(spring_line)
                
        except Exception as e:
            print(f"⚠️ Failed to create 3D vehicle: {e}")
    
    def update_simulation(self, frame):
        """Update function for animation."""
        # Get current road input
        vehicle_position = self.time * self.road_speed
        road_time = vehicle_position / self.road_speed
        current_road_input = self.current_road.get_profile(road_time)
        
        # Get control action
        if self.active_control and self.agent.Q_matrix is not None:
            control_force = self.agent.get_action(self.suspension.state, current_road_input)
        else:
            control_force = 0.0
            
        # Step simulation
        state, acceleration, _ = self.suspension.step(control_force, current_road_input)
        
        # Update history
        self.time_history.append(self.time)
        self.body_pos_history.append(state[0])
        self.wheel_pos_history.append(state[2])
        self.road_history.append(current_road_input)
        self.control_history.append(control_force)
        
        # Update 3D visualization
        vehicle_x = 10.0  # Fixed position in 3D space
        body_z = 0.1 + state[0]  # Body height above ground
        wheel_z = state[2]       # Wheel height
        
        self.create_vehicle_3d(vehicle_x, body_z, wheel_z)
        
        # Update time series plots
        self.update_time_series()
        
        # Update camera angle for dynamic view
        self.ax_3d.view_init(elev=20, azim=self.time * 10)  # Slow rotation
        
        self.time += self.dt
        
        # Return empty tuple since we're not using blitting for 3D
        return []
    
    def update_time_series(self):
        """Update 2D time series plots."""
        if len(self.time_history) < 2:
            return
            
        times = list(self.time_history)
        
        # Position plot
        self.body_line.set_data(times, list(self.body_pos_history))
        self.wheel_line.set_data(times, list(self.wheel_pos_history))
        self.road_line_plot.set_data(times, list(self.road_history))
        
        # Auto-scale position plot
        self.ax_pos.set_xlim(max(0, self.time - 5), self.time + 0.5)
        if self.body_pos_history:
            all_pos = list(self.body_pos_history) + list(self.wheel_pos_history) + list(self.road_history)
            y_min, y_max = min(all_pos), max(all_pos)
            margin = (y_max - y_min) * 0.1
            self.ax_pos.set_ylim(y_min - margin, y_max + margin)
        
        # Control plot
        self.control_line.set_data(times, list(self.control_history))
        self.ax_control.set_xlim(max(0, self.time - 5), self.time + 0.5)
        if self.control_history:
            control_range = max(abs(max(self.control_history)), abs(min(self.control_history)))
            self.ax_control.set_ylim(-control_range - 5, control_range + 5)
        
        # Add performance text
        if len(self.body_pos_history) > 10:
            recent_accel = np.diff(list(self.body_pos_history)[-20:]) / self.dt  # Approximate acceleration
            rms_accel = np.sqrt(np.mean(recent_accel**2))
            
            info_text = f"""Road: {self.current_road_name}
Control: {'Active' if self.active_control else 'Passive'}
Speed: {self.road_speed:.1f} m/s
RMS Accel: {rms_accel:.3f} m/s²
Time: {self.time:.1f} s"""
            
            # Clear and update text
            self.ax_control.texts.clear()
            self.ax_control.text(0.02, 0.98, info_text, transform=self.ax_control.transAxes,
                               fontsize=9, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def start_simulation(self):
        """Start the 3D simulation."""
        print("🚀 Starting 3D Active Suspension Simulation")
        print("🎮 Use controls to change road profiles and settings")
        print("🎥 The 3D view will slowly rotate for better perspective")
        
        # Setup plot first
        self.setup_plot()
        
        # Reset simulation
        self.suspension.reset()
        self.time = 0.0
        
        try:
            # Create animation
            self.ani = animation.FuncAnimation(
                self.fig, self.update_simulation, interval=100,  # 10 FPS for 3D
                blit=False, cache_frame_data=False
            )
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"❌ 3D Animation error: {e}")
            print("Showing static plot instead...")
            plt.show()

# Main execution
if __name__ == "__main__":
    try:
        print("Choose simulation type:")
        print("1. 2D Simulation (more detailed)")
        print("2. 3D Simulation (visual perspective)")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "2":
            sim = SuspensionSimulation3D(dt=0.001)
            sim.start_simulation()
        else:
            # Import and run 2D simulation
            from suspension_simulation import SuspensionSimulation
            sim = SuspensionSimulation(dt=0.001)
            sim.start_simulation()
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        print("Please check that all required files are available.")