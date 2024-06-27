from lab6_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.animation as anim
import matplotlib.transforms as trans

# Robot model
d = np.zeros(3)                            # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.2])          # rotation around Z-axis
alpha = np.zeros(3)                        # rotation around X-axis
a = [0.5, 0.75, 0.5]                       # displacement along X-axis
revolute = [True, True, True]              # flags specifying the type of joints
robot = MobileManipulator(d, theta, a, alpha, revolute)
weights = [0.1, 0.1, 0.1, 0.1, 0.1] #weight of each DOF
tasks = [
        Configuration2D("end-effector configuration", np.array([1.0, 0.5,0]).reshape(3,1))
        ]
# Simulation params
dt = 1.0/10.0
counter = -2
# Drawing preparation
fig = plt.figure(1)
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
rectangle = patch.Rectangle((-0.25, -0.15), 0.5, 0.3, color='blue', alpha=0.3)
veh = ax.add_patch(rectangle)
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target
PPx = []
PPy = []
time= []
error = [[],[]]
velocities =[]
EE_pos=[[],[]] # to store the ee position of each joint to be used in the second plot.
base_pos=[[],[]]
i=0


# Simulation initialization
def init():
    global tasks, i, counter
    Desired = tasks[-1].getDesired()
    desired = np.array([[-1.5, -1.5], [1.5, 1.5], [0, 0], [0.1, 1.5], [0.3, -0.1], [-0.4, 1.2]])
    Desired[0:2] = desired[i].reshape(2, 1)
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    i += 1
    counter = counter + 1
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    ### Recursive Task-Priority algorithm
    P = np.eye(robot.getDOF())
    # Initialize output vector (joint velocity)
    dq = np.zeros(robot.getDOF()).reshape(robot.getDOF(),1)

    for task in tasks:
        task.update(robot)
        if task.isActive():
            J_bar = task.getJacobian() @ P
            dq += Weighted_DLS(J_bar, 0.1, weights) @ (task.getError() - task.getJacobian() @ dq)
            P -= np.linalg.pinv(J_bar) @ J_bar
            error[0].append(np.linalg.norm(task.getError()[0:2]))
            error[1].append(np.linalg.norm(task.getError()[2]))
            velocities.append(dq)

            EE_pos[0].append(robot.getEETransform()[0, 3])
            EE_pos[1].append(robot.getEETransform()[1, 3])
            base_pos[0].append(robot.getBasePose()[0, 0])
            base_pos[1].append(robot.getBasePose()[1, 0])
            # q_base.append(robot.getBasePose()[:2,0].reshape(2,1))
    # Update robot
    robot.update(dq, dt)
    # Update drawing
    # -- Manipulator links
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    time.append(t + 10 * counter)
    path.set_data(PPx, PPy)
    point.set_data(tasks[-1].getDesired()[0], tasks[-1].getDesired()[1])
    # -- Mobile base
    eta = robot.getBasePose()
    veh.set_transform(trans.Affine2D().rotate(eta[2,0]) + trans.Affine2D().translate(eta[0,0], eta[1,0]) + ax.transData)
    return line, veh, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt),interval=10, blit=True, init_func=init, repeat=True)
plt.show()
# Plot errors over time
plt.plot(time, error[0], label='e1 (End-effector position error)')
plt.plot(time, error[1], label='e2 (End-effector orientation error)')
plt.ylabel('Error [m]')
plt.xlabel('Time [s]')
plt.title('Task-Priority Control')
plt.xlim(left=0)
plt.grid(True)
plt.legend()
plt.show()
# Save EE_pos and base_pos to file
# np.save('move forward then rotate.npy',[EE_pos,base_pos])
np.save(' rotate then move forward.npy',[EE_pos,base_pos])
# np.save(' rotate and move forward.npy',[EE_pos,base_pos])