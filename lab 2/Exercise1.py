# Import necessary libraries
from lab2_robotics import * # Import our library (includes Numpy)
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

# Robot definition (planar 2 link manipulator)
d = np.zeros(2)           # displacement along Z-axis
q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)       # rotation around X-axis 

# Simulation params
dt = 0.01 # Sampling time
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Kinematics')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.set_aspect('equal')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'r-', lw=1) # End-effector path

# Memory
PPx = []
PPy = []
q1_positions = []
q2_positions = []
# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    return line, path


# Simulation loop
def simulate(t):
    global d, q, a, alpha
    global PPx, PPy, q1_positions, q2_positions
    
    
    # Update robot
    T = kinematics(d, q, a, alpha)
    dq = np.array([0.5,0.8]) # Define how joint velocity changes with time!
    q = q + dt * dq
     # Store joint positions
    q1_positions.append(q[0])
    q2_positions.append(q[1])
    
   
    
    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    
    return line, path

# Run simulation
animation = anim.FuncAnimation(fig, simulate, tt, 
                                interval=10, blit=True, init_func=init, repeat=False)
plt.show()
# Ensure the lengths of time and position arrays are the same
min_length = min(len(tt), len(q1_positions), len(q2_positions))

tt = tt[:min_length]
q1_positions = q1_positions[:min_length]
q2_positions = q2_positions[:min_length]

# Now create the plots for joint positions over time
plt.figure()
plt.plot(tt, q1_positions, label='q1')  # Plot for q1
plt.plot(tt, q2_positions, label='q2')  # Plot for q2

# Add titles and labels
plt.title('Joint Position')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid(True)


# Display the plot of joint positions
plt.show()