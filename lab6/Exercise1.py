from lab6_robotics import *  # This imports MobileManipulator, JointLimit, Position2D classes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

# Robot model initialization
d = np.zeros(3)  # Displacement along Z-axis
theta = np.array([0.2, 0.5, 0.2])  # Rotation around Z-axis
alpha = np.zeros(3)  # Rotation around X-axis
a = [0.5, 0.75, 0.5]  # Displacement along X-axis
revolute = [True, True, True]  # All joints are revolute
robot = MobileManipulator(d, theta, a, alpha, revolute)

# Define tasks for the robot
tasks = [
    JointLimit("Joint limits", 2, np.array([-0.5, 0.5]), np.array([0.03, 0.05])),
    Position2D("End-effector position", np.array([1.0, 0.5]).reshape(2, 1))
]

# Simulation parameters
dt = 1.0 / 60.0  # Time step
count = -1  # Counter for the simulation steps

# Set up the drawing for the simulation
fig = plt.figure(1)
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2)  # Robot structure
path, = ax.plot([], [], 'c-', lw=1)  # Path of the end-effector
point, = ax.plot([], [], 'rx')  # Target point
pp_x = []  # X-coordinates of the path
pp_y = []  # Y-coordinates of the path
time = []  # Time points for plotting
errors = [[], []]  # Errors for joints and end-effector

# Initialize simulation
def init():
    global tasks, count
    count += 1
    # Set a new desired position for the end-effector
    tasks[-1].setDesired(np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)]).reshape(2, 1))
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation step function
def simulate(t):
    global robot, pp_x, pp_y
    p_matrix = np.eye(robot.getDOF())  # Projector matrix
    dq = np.zeros(robot.getDOF()).reshape(robot.getDOF(), 1)  # Differential joint velocities
    for index, task in enumerate(tasks):
        task.update(robot)
        if task.isActive():
            j_bar = task.getJacobian() @ p_matrix
            dq += DLS(j_bar, 0.1) @ (task.getError() - task.getJacobian() @ dq)
            p_matrix -= np.linalg.pinv(j_bar) @ j_bar
        errors[index].append(np.linalg.norm(task.getError()) if index else robot.getJointPos(task.joint))

    robot.update(dq, dt)
    pp = robot.drawing()
    line.set_data(pp[0, :], pp[1, :])
    pp_x.append(pp[0, -1])
    pp_y.append(pp[1, -1])
    time.append(t + 10 * count)
    path.set_data(pp_x, pp_y)
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1])
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2]) + trans.Affine2D().translate(eta[0], eta[1]) + ax.transData)
    return line, veh, path, point

# Run simulation
ani = anim.FuncAnimation(fig, simulate, frames=np.arange(0, 10, dt), interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Error plotting after simulation
fig2 = plt.figure(2)
plt.axhline(y=-0.5, color='r', linestyle='--')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.plot(time, errors[0], label='q1 (position of joint 1)')
plt.plot(time, errors[1], label='e2 (end-effector position error)')
plt.ylabel('Error [m]')
plt.xlabel('Time [s]')
plt.title('Task-Priority Control')
plt.grid(True)
plt.legend()
plt.show()
