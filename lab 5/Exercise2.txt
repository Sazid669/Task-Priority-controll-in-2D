# Importing necessary libraries
from lab4_robotics import *
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patch

# Robot model parameters
d = np.zeros(3)                            # Displacement along Z-axis
theta = np.array([0.2, 0.5, 0.2])          # Rotation around Z-axis
alpha = np.zeros(3)                        # Rotation around X-axis
a = [0.5, 0.75, 0.5]                       # Displacement along X-axis
revolute = [True, True, True]              # Flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

# List of tasks for the robot
tasks = [ 
          JointLimit("Joint limits", np.array([-0.5, 0.5]), np.array([0.01, 0.04])),  # Joint limits task with activation and deactivation thresholds 0.01 and 0.04, and safe set of q_min and q_max (-0.5 and 0.5)
          Position2D("End-effector position", np.array([0.25, -0.75]).reshape(2,1))   # End-effector position task
        ] 

# Simulation parameters
dt = 1.0/60.0
Storage = -1

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
line, = ax.plot([], [], 'o-', lw=2)  # Robot structure
path, = ax.plot([], [], 'c-', lw=1)  # End-effector path
point, = ax.plot([], [], 'rx')       # Target
# Memory
PPx = []
PPy = []
time= []
error = [[],[]]

# Simulation initialization
def init():
    global tasks, Storage

    Storage += 1
    # Setting end-effector position as final task
    tasks[len(tasks)-1].setDesired(np.array([np.random.uniform(-1.5,1.5), 
                            np.random.uniform(-1.5,1.5)]).reshape(2,1))
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point


# Main simulation loop
def simulate(t):
    global robot, PPx, PPy
    
    # Run the Recursive Task-Priority algorithm
    P = np.eye(robot.getDOF())  # Initialize the projector matrix
    dq = np.zeros(robot.getDOF()).reshape(robot.getDOF(),1)  # Initialize joint velocities

    # Loop over tasks, updating each and applying the control law
    for index, task in enumerate(tasks):
        task.update(robot)  # Update the task's internal state
        if task.isActive() != 0:
            J_bar = task.getJacobian() @ P  # Calculate the augmented Jacobian
            dq += DLS(J_bar, 0.1) @ (task.getError() - task.getJacobian() @ dq)  # Calculate the joint velocity
            P = P - np.linalg.pinv(J_bar) @ J_bar  # Update the null-space projector

        # Record the joint position or end-effector position error for plotting
        error[index].append(np.linalg.norm(task.getError()) if index else robot.getJointPos(0))

    # Update the manipulator's state
    robot.update(dq, dt)
    
    # Update the drawing for animation
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    time.append(t + 10 * Storage)
    path.set_data(PPx, PPy)
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1])
    
    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plot-2
fig2 = plt.figure(2)
# Specifying horizontal line for safe sets
plt.axhline(y = -0.5, color = 'r', linestyle = '--')
plt.axhline(y = 0.5, color = 'r', linestyle = '--')
plt.plot(time, error[0], label='q1 (position of joint 1)')  # Plotting position of joint 1 against time
plt.plot(time, error[1], label='e2 (end-effector position error)')  # Plotting position error of end-effector
plt.ylabel('Error[m]')  # Title of the Y axis
plt.xlabel('Time[s]')   # Title of the X axis
plt.title('Task-Priority control')  # Title of plot-1
plt.grid(True)  # Grid 
plt.legend()  # Placing legend
plt.show()
