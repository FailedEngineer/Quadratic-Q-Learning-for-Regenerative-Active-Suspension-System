"""
Active Suspension System - Real-Time Simulation Launcher (Fixed Version)
========================================================================

This script provides an easy way to launch and configure the suspension simulations.
Includes debugging and improved layout.
"""

# Fix OpenMP conflict BEFORE any other imports
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import subprocess

class SimulationLauncher:
    """GUI launcher for suspension simulations with improved layout."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚗 Active Suspension Simulation Launcher")
        
        # Make window larger and resizable
        self.root.geometry("700x700")
        self.root.minsize(600, 500)
        self.root.resizable(True, True)
        
        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main frame with scrolling capability
        self.create_scrollable_frame()
        
        # Check dependencies
        self.check_dependencies()
        self.setup_gui()
        
        print("✅ GUI setup completed successfully")
        
    def create_scrollable_frame(self):
        """Create a scrollable main frame."""
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def check_dependencies(self):
        """Check if all required files and packages are available."""
        print("🔍 Checking dependencies...")
        
        required_files = [
            'Suspension_Model.py',
            'Road_profile.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
                print(f"❌ Missing: {file}")
            else:
                print(f"✅ Found: {file}")
        
        if missing_files:
            error_msg = f"Missing required files:\\n{', '.join(missing_files)}\\n\\nPlease ensure all model files are in the current directory."
            messagebox.showerror("Missing Files", error_msg)
            sys.exit(1)
        
        # Check for trained agents
        self.trained_agents = self.find_trained_agents()
        print(f"🤖 Found {len(self.trained_agents)} trained agents")
        
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
        """Setup the GUI interface with improved layout."""
        print("🎨 Setting up GUI...")
        
        # Use the scrollable frame as parent
        parent = self.scrollable_frame
        
        # Title with better spacing
        title_label = tk.Label(parent, text="🚗 Active Suspension System Simulator", 
                              font=("Arial", 18, "bold"), fg="navy")
        title_label.pack(pady=(20, 10))
        print("📝 Title created")
        
        # Description with better formatting
        desc_text = """Real-time visualization of active suspension dynamics with Q-Learning control.
        
🔧 Features:
• Interactive road profile switching (Square Wave, Bumps, Rough Road)
• Active vs Passive control comparison  
• Real-time performance metrics
• Adjustable vehicle speed
• Automatic trained agent loading"""
        
        desc_label = tk.Label(parent, text=desc_text, justify=tk.LEFT, 
                             font=("Arial", 11), wraplength=650, bg="lightgray", relief="sunken")
        desc_label.pack(pady=10, padx=20, fill=tk.X)
        print("📋 Description created")
        
        # Simulation type selection with better styling
        sim_frame = tk.LabelFrame(parent, text="🎮 Simulation Type", 
                                 font=("Arial", 12, "bold"), fg="darkgreen")
        sim_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.sim_type = tk.StringVar(value="2d")
        
        tk.Radiobutton(sim_frame, text="🔍 2D Detailed View - Shows springs, dampers, and detailed metrics", 
                      variable=self.sim_type, value="2d", font=("Arial", 11)).pack(anchor=tk.W, padx=15, pady=8)
        
        tk.Radiobutton(sim_frame, text="🌐 3D Perspective View - Shows vehicle in 3D space with rotating camera", 
                      variable=self.sim_type, value="3d", font=("Arial", 11)).pack(anchor=tk.W, padx=15, pady=8)
        print("🎮 Simulation type options created")
        
        # Configuration options with grid layout
        config_frame = tk.LabelFrame(parent, text="⚙️ Configuration", 
                                   font=("Arial", 12, "bold"), fg="darkblue")
        config_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Animation Speed
        tk.Label(config_frame, text="Animation Speed:", font=("Arial", 11)).grid(row=0, column=0, sticky=tk.W, padx=15, pady=8)
        self.timestep = tk.StringVar(value="Normal (dt=0.01)")
        timestep_combo = ttk.Combobox(config_frame, textvariable=self.timestep, 
                                     values=["model accurate (dt=0.001)", "Normal (dt=0.01)", "Slow (dt=0.02)"],
                                     state="readonly", width=25)
        timestep_combo.grid(row=0, column=1, padx=15, pady=8, sticky=tk.W)
        
        # Initial road profile
        tk.Label(config_frame, text="Starting Road Profile:", font=("Arial", 11)).grid(row=1, column=0, sticky=tk.W, padx=15, pady=8)
        self.road_profile = tk.StringVar(value="Square Wave")
        road_combo = ttk.Combobox(config_frame, textvariable=self.road_profile,
                                 values=["Square Wave", "Large Bump", "Small Bump", "Rough Road"],
                                 state="readonly", width=25)
        road_combo.grid(row=1, column=1, padx=15, pady=8, sticky=tk.W)
        print("⚙️ Configuration options created")
        
        # Agent status with better formatting
        agent_frame = tk.LabelFrame(parent, text="🤖 Trained Agent Status", 
                                   font=("Arial", 12, "bold"), fg="purple")
        agent_frame.pack(pady=10, padx=20, fill=tk.X)
        
        if self.trained_agents:
            agent_text = f"✅ Found {len(self.trained_agents)} trained agent(s):\\n"
            for agent in self.trained_agents[:3]:  # Show max 3
                agent_text += f"   • {agent}\\n"
            if len(self.trained_agents) > 3:
                agent_text += f"   • ... and {len(self.trained_agents) - 3} more"
        else:
            agent_text = "⚠️ No trained agents found. Simulation will run in passive mode only."
        
        tk.Label(agent_frame, text=agent_text, justify=tk.LEFT, font=("Arial", 10), 
                bg="lightyellow").pack(padx=15, pady=10, fill=tk.X)
        print("🤖 Agent status created")
        
        # CONTROL BUTTONS - This is the main part that was missing!
        print("🔘 Creating control buttons...")
        
        # Create button frame with better styling
        button_frame = tk.Frame(parent, bg="lightblue", relief="raised", bd=2)
        button_frame.pack(pady=30, padx=20, fill=tk.X)
        print("📦 Button frame created")
        
        # Title for button section
        button_title = tk.Label(button_frame, text="🚀 Launch Controls", 
                               font=("Arial", 14, "bold"), bg="lightblue")
        button_title.pack(pady=(10, 5))
        
        # Button container
        btn_container = tk.Frame(button_frame, bg="lightblue")
        btn_container.pack(pady=10)
        
        # Launch Simulation Button
        launch_btn = tk.Button(btn_container, text="🚀 Launch Simulation", 
                              command=self.launch_simulation, font=("Arial", 12, "bold"),
                              bg="#4CAF50", fg="white", padx=30, pady=15,
                              relief="raised", bd=3, cursor="hand2")
        launch_btn.pack(side=tk.LEFT, padx=15)
        print("✅ Launch button created")
        
        # Train New Agent Button
        train_btn = tk.Button(btn_container, text="🎯 Train New Agent", 
                             command=self.launch_training, font=("Arial", 12, "bold"),
                             bg="#2196F3", fg="white", padx=30, pady=15,
                             relief="raised", bd=3, cursor="hand2")
        train_btn.pack(side=tk.LEFT, padx=15)
        print("✅ Train button created")
        
        # Quit Button
        quit_btn = tk.Button(btn_container, text="❌ Quit", command=self.safe_quit,
                            font=("Arial", 12, "bold"), bg="#f44336", fg="white", 
                            padx=30, pady=15, relief="raised", bd=3, cursor="hand2")
        quit_btn.pack(side=tk.LEFT, padx=15)
        print("✅ Quit button created")
        
        # Instructions with better formatting
        instructions_frame = tk.LabelFrame(parent, text="📖 Simulation Controls", 
                                         font=("Arial", 12, "bold"), fg="darkorange")
        instructions_frame.pack(pady=15, padx=20, fill=tk.X)
        
        instructions_text = """During simulation:
🎮 Click road profile buttons to change road conditions in real-time
🔄 Toggle 'Active Control' to compare active vs passive suspension  
⚡ Adjust speed slider to change vehicle velocity
📊 Watch performance metrics update in real-time
❌ Close the plot window to return to this launcher"""
        
        tk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT, 
                font=("Arial", 10), wraplength=650, bg="lightyellow").pack(padx=15, pady=10)
        print("📖 Instructions created")
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to launch simulation")
        status_bar = tk.Label(parent, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W, font=("Arial", 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 20))
        print("📊 Status bar created")
        
        print("🎨 GUI setup completed!")
    
    def safe_quit(self):
        """Safely quit the application."""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.root.quit()
    
    def get_timestep(self):
        """Get timestep value from selection."""
        timestep_map = {
            "model accurate (dt=0.001)": 0.001,
            "Normal (dt=0.01)": 0.01,
            "Slow (dt=0.02)": 0.02
        }
        return timestep_map.get(self.timestep.get(), 0.01)
    
    def launch_simulation(self):
        """Launch the selected simulation."""
        self.status_var.set("Launching simulation...")
        
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
                
            self.status_var.set("Simulation completed")
        except Exception as e:
            error_msg = f"Failed to launch simulation:\\n{str(e)}"
            messagebox.showerror("Launch Error", error_msg)
            self.status_var.set("Launch failed")
            print(f"❌ Launch error: {e}")
    
    def launch_2d_simulation(self, dt, road_profile):
        """Launch 2D simulation."""
        self.root.withdraw()  # Hide launcher window
        
        try:
            # Check if simulation file exists
            if os.path.exists("suspension_simulation.py"):
                print("📂 Found suspension_simulation.py, importing...")
                from suspension_simulation import SuspensionSimulation
                sim = SuspensionSimulation(dt=dt)
                sim.change_road_profile(road_profile)
                sim.start_simulation()
            else:
                messagebox.showinfo("File Not Found", 
                                  "suspension_simulation.py not found.\\n"
                                  "Please ensure the file is in the current directory.")
        except ImportError as e:
            messagebox.showerror("Import Error", f"Failed to import simulation:\\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Simulation failed:\\n{str(e)}")
        finally:
            self.root.deiconify()  # Show launcher window again
    
    def launch_3d_simulation(self, dt, road_profile):
        """Launch 3D simulation."""
        self.root.withdraw()  # Hide launcher window
        
        try:
            # Check if simulation file exists
            if os.path.exists("suspension_3d_simulation.py"):
                print("📂 Found suspension_3d_simulation.py, importing...")
                from suspension_3d_simulation import SuspensionSimulation3D
                sim = SuspensionSimulation3D(dt=dt)
                sim.change_road_profile(road_profile)
                sim.start_simulation()
            else:
                messagebox.showinfo("File Not Found", 
                                  "suspension_3d_simulation.py not found.\\n"
                                  "Please ensure the file is in the current directory.")
        except ImportError as e:
            messagebox.showerror("Import Error", f"Failed to import 3D simulation:\\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Simulation Error", f"3D Simulation failed:\\n{str(e)}")
        finally:
            self.root.deiconify()  # Show launcher window again
    
    def launch_training(self):
        """Launch the training script."""
        if os.path.exists("Qlearning_Training.py"):
            result = messagebox.askyesno("Launch Training", 
                                       "This will start Q-Learning training which may take several hours.\\n\\n"
                                       "Do you want to continue?")
            if result:
                try:
                    subprocess.Popen([sys.executable, "Qlearning_Training.py"])
                    messagebox.showinfo("Training Started", "Training has been launched in a separate process.")
                    self.status_var.set("Training launched in background")
                except Exception as e:
                    messagebox.showerror("Training Error", f"Failed to launch training:\\n{str(e)}")
                    self.status_var.set("Training launch failed")
        else:
            messagebox.showerror("File Not Found", "Qlearning_Training.py not found in current directory.")
    
    def run(self):
        """Run the launcher."""
        print("🎮 Active Suspension Simulation Launcher")
        print("=" * 50)
        print("Starting GUI launcher...")
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (700 // 2)
        y = (self.root.winfo_screenheight() // 2) - (700 // 2)
        self.root.geometry(f'700x700+{x}+{y}')
        
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
        print("🖥️ Starting GUI launcher...")
        launcher = SimulationLauncher()
        launcher.run()
    except ImportError:
        print("⚠️ GUI not available. Running quick demo instead...")
        create_quick_demo()
    except Exception as e:
        print(f"❌ GUI Error: {e}")
        print("Running quick demo instead...")
        create_quick_demo()