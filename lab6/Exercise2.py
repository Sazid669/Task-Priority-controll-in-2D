# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans
from lab6_robotics import *

# Define constants and parameters
dt = 1.0 / 60.0
counter = -1
weights = [2, 4, 6, 8, 9]

# Initialize lists for storing data
PPx = []
PPy = []
time = []
error = [[], []]
velocities = []

# Initialize simulation figure
fig = plt.figure(1)
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')

# Initialize robot model
d = np.zeros(3)
theta = np.array([0.2, 0.5, 0.2])
alpha = np.zeros(3)
a = [0.5, 0.75, 0.5]
revolute = [True, True, True]
robot = MobileManipulator(d, theta, a, alpha, revolute)

# Define tasks
tasks = [
    Configuration2D("end-effector configuration", np.array([1.0, 0.5, 0]).reshape(3, 1))
]

# Define drawing elements
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2)  # Robot structure
path, = ax.plot([], [], 'c-', lw=1)  # End-effector path
point, = ax.plot([], [], 'rx')  # Target

# Initialize animation
def init():
    global tasks, counter
    counter += 1
    tasks[-1].setDesired(np.array([np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5), 0.2]).reshape(3, 1))
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Run simulation loop
def simulate(t):
    global tasks, robot, PPx, PPy, time, error, velocities
    P = np.eye(robot.getDOF())
    dq = np.zeros(robot.getDOF()).reshape(robot.getDOF(), 1)
    for task in tasks:
        task.update(robot)
        if task.isActive():
            J_bar = task.getJacobian() @ P
            dq += Weighted_DLS(J_bar, 0.1, weights) @ (task.getError() - task.getJacobian() @ dq)
            P -= np.linalg.pinv(J_bar) @ J_bar
            error[0].append(np.linalg.norm(task.getError()[0:2]))
            error[1].append(np.linalg.norm(task.getError()[2]))
            velocities.append(dq)
    robot.update(dq, dt)
    PP = robot.drawing()
    line.set_data(PP[0, :], PP[1, :])
    PPx.append(PP[0, -1])
    PPy.append(PP[1, -1])
    time.append(t + 10 * counter)
    path.set_data(PPx, PPy)
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1])
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2, 0]) + trans.Affine2D().translate(eta[0, 0], eta[1, 0]) + ax.transData)
    return line, veh, path, point

# Initialize and run animation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plot for end-effector configuration task error
fig, ax1 = plt.subplots()
fig.suptitle('End-effector Configuration Task Error')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Error')
ax1.plot(time, error[0], alpha=0.6, linewidth=2, label='e1 (End-effector position)')
ax1.plot(time, error[1], alpha=0.6, linewidth=2, label='e2 (End-effector orientation)')
ax1.legend()
ax1.grid()
plt.show()

# Plot for velocities of all controlled DOF of the robot
fig, ax2 = plt.subplots()
fig.suptitle('Velocity of All Controlled DOF of the Robot')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Velocity [m/s]')
ax2.plot(time, [v[0, 0] for v in velocities], label='e1 (velocity of joint 1)')
ax2.plot(time, [v[1, 0] for v in velocities], label='e2 (velocity of joint 2)')
ax2.plot(time, [v[2, 0] for v in velocities], label='e3 (velocity of joint 3)')
ax2.plot(time, [v[3, 0] for v in velocities], label='e4 (velocity of joint 4)')
ax2.plot(time, [v[4, 0] for v in velocities], label='e5 (velocity of joint 5)')
ax2.legend()
ax2.grid()
plt.show()


plt.show()
