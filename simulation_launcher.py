"""
Active Suspension System - Real-Time Simulation Launcher
========================================================

This script provides an easy way to launch and configure the suspension simulations.
It includes both 2D detailed and 3D perspective views of the active suspension system.

Features:
- Real-time visualization of suspension dynamics
- Interactive road profile switching
- Active/Passive control comparison
- Performance metrics display
- Adjustable vehicle speed
- Automatic trained agent loading

Usage:
    python simulation_launcher.py

Requirements:
    - matplotlib
    - numpy
    - torch (for trained agent loading)
    - Your suspension model files (Suspension_Model.py, Road_profile.py)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import messagebox, ttk
import subprocess

class SimulationLauncher:
    """GUI launcher for suspension simulations."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Active Suspension Simulation Launcher")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Check dependencies
        self.check_dependencies()
        self.setup_gui()
        
    def check_dependencies(self):
        """Check if all required files and packages are available."""
        required_files = [
            'Suspension_Model.py',
            'Road_profile.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            error_msg = f"Missing required files:\n{', '.join(missing_files)}\n\nPlease ensure all model files are in the current directory."
            messagebox.showerror("Missing Files", error_msg)
            sys.exit(1)
        
        # Check for trained agents
        self.trained_agents = self.find_trained_agents()
        
    def find_trained_agents(self):
        """Find available trained agents."""
        agents = []
        
        if os.path.exists("INTERRUPTED_agent.npy"):
            agents.append("INTERRUPTED_agent.npy (Latest Interrupted)")
        if os.path.exists("trained_suspension_agent_v3.npy"):
            agents.append("trained_suspension_agent_v3.npy (Final)")
        if os.path.isdir("checkpoints"):
            import glob
            checkpoints = glob.glob("checkpoints/checkpoint_episode_*.npy")
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
                agents.append(f"{latest_checkpoint} (Checkpoint)")
        
        return agents
    
    def setup_gui(self):
        """Setup the GUI interface."""
        # Title
        title_label = tk.Label(self.root, text="🚗 Active Suspension System Simulator", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Description
        desc_text = """Real-time visualization of active suspension dynamics with Q-Learning control.
        
Features:
• Interactive road profile switching (Square Wave, Bumps, Rough Road)
• Active vs Passive control comparison
• Real-time performance metrics
• Adjustable vehicle speed
• Automatic trained agent loading"""
        
        desc_label = tk.Label(self.root, text=desc_text, justify=tk.LEFT, 
                             font=("Arial", 10), wraplength=550)
        desc_label.pack(pady=10)
        
        # Simulation type selection
        sim_frame = tk.LabelFrame(self.root, text="Simulation Type", font=("Arial", 12, "bold"))
        sim_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.sim_type = tk.StringVar(value="2d")
        
        tk.Radiobutton(sim_frame, text="🔍 2D Detailed View - Shows springs, dampers, and detailed metrics", 
                      variable=self.sim_type, value="2d", font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=5)
        
        tk.Radiobutton(sim_frame, text="🌐 3D Perspective View - Shows vehicle in 3D space with rotating camera", 
                      variable=self.sim_type, value="3d", font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=5)
        
        # Configuration options
        config_frame = tk.LabelFrame(self.root, text="Configuration", font=("Arial", 12, "bold"))
        config_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Time step
        tk.Label(config_frame, text="Animation Speed:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.timestep = tk.StringVar(value="Normal")
        timestep_combo = ttk.Combobox(config_frame, textvariable=self.timestep, 
                                     values=["Fast (dt=0.005)", "Normal (dt=0.01)", "Slow (dt=0.02)"],
                                     state="readonly", width=20)
        timestep_combo.grid(row=0, column=1, padx=10, pady=5)
        
        # Initial road profile
        tk.Label(config_frame, text="Starting Road Profile:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.road_profile = tk.StringVar(value="Square Wave")
        road_combo = ttk.Combobox(config_frame, textvariable=self.road_profile,
                                 values=["Square Wave", "Large Bump", "Small Bump", "Rough Road"],
                                 state="readonly", width=20)
        road_combo.grid(row=1, column=1, padx=10, pady=5)
        
        # Agent status
        agent_frame = tk.LabelFrame(self.root, text="Trained Agent Status", font=("Arial", 12, "bold"))
        agent_frame.pack(pady=10, padx=20, fill=tk.X)
        
        if self.trained_agents:
            agent_text = f"✅ Found {len(self.trained_agents)} trained agent(s):\n"
            for agent in self.trained_agents[:3]:  # Show max 3
                agent_text += f"   • {agent}\n"
            if len(self.trained_agents) > 3:
                agent_text += f"   • ... and {len(self.trained_agents) - 3} more"
        else:
            agent_text = "⚠️ No trained agents found. Simulation will run in passive mode only."
        
        tk.Label(agent_frame, text=agent_text, justify=tk.LEFT, font=("Arial", 9)).pack(padx=10, pady=5)
        
        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        launch_btn = tk.Button(button_frame, text="🚀 Launch Simulation", 
                              command=self.launch_simulation, font=("Arial", 12, "bold"),
                              bg="#4CAF50", fg="white", padx=20, pady=10)
        launch_btn.pack(side=tk.LEFT, padx=10)
        
        train_btn = tk.Button(button_frame, text="🎯 Train New Agent", 
                             command=self.launch_training, font=("Arial", 12),
                             bg="#2196F3", fg="white", padx=20, pady=10)
        train_btn.pack(side=tk.LEFT, padx=10)
        
        quit_btn = tk.Button(button_frame, text="❌ Quit", command=self.root.quit,
                            font=("Arial", 12), bg="#f44336", fg="white", padx=20, pady=10)
        quit_btn.pack(side=tk.LEFT, padx=10)
        
        # Instructions
        instructions_frame = tk.LabelFrame(self.root, text="Simulation Controls", font=("Arial", 10, "bold"))
        instructions_frame.pack(pady=10, padx=20, fill=tk.X)
        
        instructions_text = """During simulation:
• Click road profile buttons to change road conditions in real-time
• Toggle 'Active Control' to compare active vs passive suspension
• Adjust speed slider to change vehicle velocity
• Watch performance metrics update in real-time
• Close the plot window to return to this launcher"""
        
        tk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT, 
                font=("Arial", 9), wraplength=550).pack(padx=10, pady=5)
    
    def get_timestep(self):
        """Get timestep value from selection."""
        timestep_map = {
            "Fast (dt=0.005)": 0.005,
            "Normal (dt=0.01)": 0.01,
            "Slow (dt=0.02)": 0.02
        }
        return timestep_map.get(self.timestep.get(), 0.01)
    
    def launch_simulation(self):
        """Launch the selected simulation."""
        sim_type = self.sim_type.get()
        dt = self.get_timestep()
        road_profile = self.road_profile.get()
        
        print(f"🚀 Launching {sim_type.upper()} simulation...")
        print(f"   Timestep: {dt}s")
        print(f"   Initial Road: {road_profile}")
        
        try:
            if sim_type == "3d":
                self.launch_3d_simulation(dt, road_profile)
            else:
                self.launch_2d_simulation(dt, road_profile)
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch simulation:\n{str(e)}")
    
    def launch_2d_simulation(self, dt, road_profile):
        """Launch 2D simulation."""
        # Hide launcher window
        self.root.withdraw()
        
        try:
            # Import and create 2D simulation
            from suspension_simulation import SuspensionSimulation
            sim = SuspensionSimulation(dt=dt)
            sim.change_road_profile(road_profile)
            sim.start_simulation()
        except ImportError:
            # If import fails, create inline
            self.create_inline_2d_simulation(dt, road_profile)
        
        # Show launcher window again after simulation closes
        self.root.deiconify()
    
    def launch_3d_simulation(self, dt, road_profile):
        """Launch 3D simulation."""
        # Hide launcher window
        self.root.withdraw()
        
        try:
            # Import and create 3D simulation
            from suspension_3d_simulation import SuspensionSimulation3D
            sim = SuspensionSimulation3D(dt=dt)
            sim.change_road_profile(road_profile)
            sim.start_simulation()
        except ImportError:
            # If import fails, create inline
            self.create_inline_3d_simulation(dt, road_profile)
        
        # Show launcher window again after simulation closes
        self.root.deiconify()
    
    def create_inline_2d_simulation(self, dt, road_profile):
        """Create 2D simulation inline if import fails."""
        # This would contain the full 2D simulation code
        # For brevity, showing a simplified version
        messagebox.showinfo("Info", "2D Simulation code would run here.\nPlease ensure suspension_simulation.py is available.")
    
    def create_inline_3d_simulation(self, dt, road_profile):
        """Create 3D simulation inline if import fails."""
        # This would contain the full 3D simulation code
        # For brevity, showing a simplified version
        messagebox.showinfo("Info", "3D Simulation code would run here.\nPlease ensure suspension_3d_simulation.py is available.")
    
    def launch_training(self):
        """Launch the training script."""
        if os.path.exists("Qlearning_Training.py"):
            result = messagebox.askyesno("Launch Training", 
                                       "This will start Q-Learning training which may take several hours.\n\n"
                                       "Do you want to continue?")
            if result:
                try:
                    subprocess.Popen([sys.executable, "Qlearning_Training.py"])
                    messagebox.showinfo("Training Started", "Training has been launched in a separate process.")
                except Exception as e:
                    messagebox.showerror("Training Error", f"Failed to launch training:\n{str(e)}")
        else:
            messagebox.showerror("File Not Found", "Qlearning_Training.py not found in current directory.")
    
    def run(self):
        """Run the launcher."""
        print("🎮 Active Suspension Simulation Launcher")
        print("=" * 50)
        print("Starting GUI launcher...")
        self.root.mainloop()

def create_quick_demo():
    """Create a quick demo without the full GUI."""
    print("🎮 Quick Suspension Demo")
    print("=" * 30)
    
    try:
        from Suspension_Model import QuarterCarModel
        from Road_profile import SquareWaveProfile
        
        # Simple command-line demo
        print("Running a quick 2-second simulation...")
        
        model = QuarterCarModel(dt=0.01)
        road = SquareWaveProfile(period=2.0, amplitude=0.02)
        
        times = []
        positions = []
        
        for i in range(200):  # 2 seconds
            t = i * 0.01
            road_input = road.get_profile(t)
            state, accel, power = model.step(0, road_input)  # Passive mode
            times.append(t)
            positions.append(state[0])
        
        # Simple plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(times, positions, label='Body Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Quick Suspension Demo (Passive Mode)')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        print("✅ Demo completed! Launch the full simulation for interactive features.")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please ensure all model files are available.")

if __name__ == "__main__":
    # Check if GUI is available
    try:
        import tkinter
        launcher = SimulationLauncher()
        launcher.run()
    except ImportError:
        print("⚠️ GUI not available. Running quick demo instead...")
        create_quick_demo()