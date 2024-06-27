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
robot = Manipulator(d, theta, a, alpha, revolute)  # Instantiate the manipulator

# Set up tasks for obstacle avoidance and end-effector targeting
obstacle_pos = np.array([0.0, 1.0]).reshape(2, 1)  # Position of the first obstacle
obstacle_r = 0.5  # Radius of the first obstacle
obstacle_pos2 = np.array([0.5, -1.3]).reshape(2, 1)  # Position of the second obstacle
obstacle_r2 = 0.6  # Radius of the second obstacle
obstacle_pos3 = np.array([-1.5, -0.5]).reshape(2, 1)  # Position of the third obstacle
obstacle_r3 = 0.4  # Radius of the third obstacle

tasks = [
    Obstacle2D("Obstacle avoidance", obstacle_pos, np.array([obstacle_r, obstacle_r + 0.05])),
    Obstacle2D("Obstacle avoidance", obstacle_pos2, np.array([obstacle_r2, obstacle_r2 + 0.05])),
    Obstacle2D("Obstacle avoidance", obstacle_pos3, np.array([obstacle_r3, obstacle_r3 + 0.05])),
    Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2, 1))
]

# Define simulation parameters
dt = 1.0 / 60.0  # Time step for simulation
Storage = -1  # Index used for storing simulation data across runs

# Set up the visualization environment
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
# Visualize obstacles as patches
ax.add_patch(patch.Circle(obstacle_pos.flatten(), obstacle_r, color='red', alpha=0.5))
ax.add_patch(patch.Circle(obstacle_pos2.flatten(), obstacle_r2, color='blue', alpha=0.5))
ax.add_patch(patch.Circle(obstacle_pos3.flatten(), obstacle_r3, color='green', alpha=0.5))

# Elements for animation
line, = ax.plot([], [], 'o-', lw=2)  # Visual representation of the robot
path, = ax.plot([], [], 'c-', lw=1)  # Path of the end-effector
point, = ax.plot([], [], 'rx')       # Desired target position for the end-effector

# Data storage for plotting
PPx, PPy = [], []  # Path position storage
time = []  # Time storage for x-axis in plots
error = [[], [], [], []]  # Error storage for the different tasks

# Initialize the simulation
def init():
    global tasks, Storage
    Storage += 1

    # Set a random desired position for the end-effector task
    tasks[-1].setDesired(np.array([np.random.uniform(-1.5, 1.5),
                                    np.random.uniform(-1.5, 1.5)]).reshape(2, 1))

    # Clear data from previous frames
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop called by the animation
# Simulation loop called by the animation
def simulate(t):
    global tasks, robot, PPx, PPy

    # Use a recursive task-priority framework for control
    P = np.eye(robot.getDOF())  # Null-space projector
    dq = np.zeros(robot.getDOF()).reshape(robot.getDOF(), 1)  # Initialize joint velocities

    for index, task in enumerate(tasks):  # Iterate over each task
        task.update(robot)  # Update task information based on current robot state

        if task.isActive():  # Check if the task is active
            J_bar = task.getJacobian() @ P  # Compute the dynamically consistent inverse
            dq += DLS(J_bar, 0.1) @ (task.getError() - task.getJacobian() @ dq)  # Compute the joint velocity increment
            P -= np.linalg.pinv(J_bar) @ J_bar  # Update the projector

        # Record errors for plotting. Use getError() for end-effector task and distance for obstacle tasks
        
        if task.name == "End-effector position":
            error[index].append(np.linalg.norm(task.getError()))
        else:
            #distance form obstacles
            error[index].append(np.linalg.norm(task.distance))
    # Update robot state with computed joint velocities
    robot.update(dq, dt)

    # Drawing updates
    PP = robot.drawing()
    line.set_data(PP[0, :], PP[1, :])
    PPx.append(PP[0, -1])
    PPy.append(PP[1, -1])
    time.append(t + 10 * Storage)
    path.set_data(PPx, PPy)
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1])

    return line, path, point


# Create the animation with the simulate function as the animation driver
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt),
                               interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plot the errors of each task over time after simulation
fig = plt.figure()
plt.plot(time, error[0], label='d1 (distance to obstacle 1)')
plt.plot(time, error[1], label='d2 (distance to obstacle 2)')
plt.plot(time, error[2], label='d3 (distance to obstacle 3)')
plt.plot(time, error[3], label='e1 (end-effector position error)')
plt.ylabel('Error [m]')
plt.xlabel('Time [s]')
plt.title('Task-Priority Inequality Tasks Performance')
plt.grid(True)
plt.legend()
plt.show()
