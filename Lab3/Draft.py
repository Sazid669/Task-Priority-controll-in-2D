# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition (3 revolute joint planar manipulator)
d = np.zeros(3)                                     # displacement along Z-axis
q = np.array([0.2, 0.5, 0.2]).reshape(3,1)                     # rotation around Z-axis (theta)
a = np.array([0.75, 0.5, 0.5])                      # displacement along X-axis
alpha = np.zeros(3)                                 # rotation around X-axis 
revolute = [True, True, True]                       # flags specifying the type of joints
k=np.eye(2)                                         # gain
max_velocity = 0.5

# Desired values of task variables
# sigma1_d = np.array([0.0, 1.0]).reshape(2,1) # Position of the end-effector
sigma2_d = np.array([[0.0]]) # Position of joint 1

# Simulation params
dt = 1.0/60.0
Tt = 10 # Total simulation time
tt = np.arange(0, Tt, dt) # Simulation time vector
count = -1 # for the loop

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
ax.grid()
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target

# storing data 
PPx = []
PPy = []
time = []
error1 = []
error2 = []

# Simulation initialization
def init():
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global q, a, d, alpha, revolute, sigma1_d, sigma2_d, max_velocity
    global PPx, PPy, count
    
    # random position of the end-effector
    if t == 0:
        sigma1_d = np.array([np.random.uniform(-1.5,1.5),np.random.uniform(-1.5,1.5)]).reshape(2,1) 
        count = count + 1
    
    # Update robot
    T = kinematics(d, q.flatten(), a, alpha)
    J = jacobian(T, revolute)
    #<<<<<<<<<<<<<<<<<<End-effector position task at the top of the hierarchy.>>>>>>>>>>>>>>>>>>>>>>
    # Update control
    # TASK 1
    sigma1 = T[-1][:2,3].reshape(2,1)        # Current position of the end-effector
    err1 =  sigma1_d - sigma1                 # Error in Cartesian position
    J1 =  J[:2,:3]                          # Jacobian of the first task
    P1 =  np.eye(3) - np.linalg.pinv(J1) @ J1 # Null space projector
    
    # TASK 2
    sigma2 = q[0]                # Current position of joint 1
    err2 = sigma2_d - sigma2     # Error in joint position
    J2 =  np.array([[0, 1, 0]])  # Jacobian of the second task
    J2bar = J2 @ P1              # Augmented Jacobian
    # Combining tasks
    dq1 =  DLS(J1,0.1) @ err1      # Velocity for the first task
    dq12 = dq1 + DLS(J2bar, 0.1) @ (err2 - J2 @ dq1)   # Velocity for both tasks
    
    #velocity scalling
    s = np.max(dq12/max_velocity)
    if s > 1:
        dq12 = dq12/s
    
    q = q + dq12 * dt # Simulation update
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Joint position task at the top of the hierarchy>>>>>>>>>>>>>>>>>>>
    # # Update control
    # # TASK 1
    # sigma1 = q[1]                # Current position of joint 1
    # err1 = sigma2_d - sigma1     # Error in joint position
    # J1 =  np.array([[0, 1, 0]])  # Jacobian of the second task
    # P1 =  np.eye(3) - np.linalg.pinv(J1) @ J1 # Null space projector
    
    
    # # TASK 2
    # sigma2 = T[-1][0:2,3].reshape(2,1)        # Current position of the end-effector
    # err2 =  sigma1_d - sigma2                 # Error in Cartesian position
    # J2 =  J[:2,:3]                            # Jacobian of the first task
    # J2bar = J2 @ P1                           # Augmented Jacobian
    
    # # Combining tasks
    # dq1 =  DLS(J1,0.1) @ err1      # Velocity for the first task
    # dq12 = dq1 + DLS(J2bar, 0.1) @ (err2 - J2 @ dq1)   # Velocity for both tasks
    
    # #velocity scalling
    # s = np.max(dq12/max_velocity)
    # if s > 1:
    #     dq12 = dq12/s
    
    # q = q + dq12 * dt # Simulation update
    # Update drawing
    PP = robotPoints2D(T)
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    path.set_data(PPx, PPy)
    point.set_data(sigma1_d[0], sigma1_d[1])
    time.append(t + 10 * count)
    error1.append(np.linalg.norm(err1))
    error2.append(np.linalg.norm(err2))

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

# Plotting simulation
fig = plt.figure()
# for end effector in the task 1
plt.plot(time, error1, label = 'e1(end effector position)')
plt.plot(time, error2, label = 'e2(Joint 1 position)' )
# for end effector in the task 2 and joint-1 in task 1
# plt.plot(time, error1, label='e1 (Joint 1 position)') 
# plt.plot(time, error2, label='e2 (End-effector position)') 
plt.ylabel('Error[m]') #Title of the Y axis
plt.ylabel('error[m]')
plt.xlabel('time[s]')
plt.title('Task Priority')
plt.grid(True)
plt.legend()
plt.show()