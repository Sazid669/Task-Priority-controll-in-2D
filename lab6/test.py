# Let's start by transcribing the code from the screenshots into a Python script.

# First, we import necessary libraries.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

# Including the library that seems to be a custom library for the robot simulation.
# As we don't have access to this library in the PCI, we'll proceed as if we do.
from lab6_robotics import *  # This includes numpy import

# Define robot model parameters
d = np.zeros(3)  # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.1])  # rotation around Z-axis
alpha = np.zeros(3)  # rotation around X-axis
a = np.array([0.75, 0.5, 0.45])  # displacement along X-axis
revolute = [True, True, True]  # flags specifying the type of joints

# Instantiate the robot
robot = MobileManipulator(d, theta, a, alpha, revolute)

# Task hierarchy definition
thresholds = np.array([0.03, 0.05])  # activation and deactivation thresholds
safe_set = np.array([-0.5, 0.5])  # safe_set: q_min, q_max

# Task definition
tasks = [
    JointLimit("Joint limits", 2, safe_set, thresholds),
    Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2, 1),5)
]

# Simulation parameters
dt = 1.0 / 60.0
counter = -1

# Drawing preparation
fig = plt.figure(1)
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')

# Drawing elements
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2)  # Robot structure
path, = ax.plot([], [], 'c-', lw=1)  # End-effector path
point, = ax.plot([], [], 'rx')  # Target

PPx = []  # List to store x coordinates for the path
PPy = []  # List to store y coordinates for the path
time = []  # List to store time points
error = [[], []]  # List to store errors

# Simulation initialization function
def init():
    global tasks, counter
    counter += 1

    # Update the final task position randomly
    tasks[len(tasks) - 1].setDesired(np.array([np.random.uniform(-1.5, 1.5),
                                                np.random.uniform(-1.5, 1.5)]).reshape(2, 1))

    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    
    return line, path, point

# Simulation loop function
def simulate(t):
    global tasks, robot, PPx, PPy

    P = np.eye(robot.getDOF())  # Initialize the identity matrix of size DOF

    # Initial output vector for joint velocities
    dq = np.zeros(robot.getDOF()).reshape(robot.getDOF(), 1)

    count = 0  # Task counter
    for i in tasks:  # Looping over tasks
        i.update(robot)  # Update task state
        if i.isActive():
            J_bar = i.getJacobian() @ P  # Compute augmented Jacobian
            # Compute and accumulate task velocity
            dq += dq + DLS(J_bar, 0.1) @ (i.getError() - i.getJacobian() @ dq)
            P = P - np.linalg.pinv(J_bar) @ J_bar  # Update null-space projector
        
        # Appending joint position or end-effector position error
        if count == 0:
            error[count].append(robot.getJointPos(i.joint))
        else:
            error[count].append(np.linalg.norm(i.getError()))
        
        count += 1

    # Update robot state
    robot.update(dq, dt)

    # Update the drawing
    PP = robot.drawing()
    line.set_data(PP[0, :], PP[1, :])
    PPy.append(PP[1, -1])
    PPx.append(PP[0, -1])
    time.append(t + 10 * counter)

    path.set_data(PPx, PPy)
    # Continuing from where the code was interrupted.
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1])

    # Get robot base pose for the visualization
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax.transData)

    return line, veh, path, point

# Run the simulation using matplotlib's animation function
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), interval=10, blit=True, init_func=init, repeat=True)

# Show the first plot
plt.show()

# Second plot for displaying horizontal lines for safe sets
fig2 = plt.figure(2)
plt.axhline(y=-0.5, color='r', linestyle='--')  # Lower safe set limit
plt.axhline(y=0.5, color='r', linestyle='--')  # Upper safe set limit
# Now, let's focus on the plotting of errors over time from the first screenshot.
plt.plot(time, error[0], label='q1 (position of joint 1)')  # Plotting position of joint 1 against time
plt.plot(time, error[1], label='e2 (end-effector position error)')  # Plotting position error of end-effector against time
plt.ylabel('Error [m]')  # Title of the Y axis
plt.xlabel('Time [s]')  # Title of the X axis
plt.title('Task-Priority control')  

plt.grid(True)  # Grid
plt.legend()  # Placing legend
plt.show()  # Show the plot

# The final code is now a complete script based on the screenshots provided.
# Next step is to save this
