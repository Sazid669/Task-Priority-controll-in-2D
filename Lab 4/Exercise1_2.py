from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot model - 3-link manipulator
d = np.zeros(3)                                 # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.2])                       # rotation around Z-axis
alpha = np.zeros(3)                        # rotation around X-axis
a = [0.5, 0.75, 0.5]                            # displacement along X-axis
revolute = [True, True, True]                     # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object
max_velocity = 0.5
# Task hierarchy definition
tasks = [   #<<<<<<<<<<<<<<<Exercise-1>>>>>>>>>>>>
            # Position2D("End-effector position", np.array([1.0,0.5]).reshape(2,1),0,0,0),
            # Orientation2D("End-effector orientation", np.array([0]).reshape(1,1),0,0,0),
            # Configuration2D("End-effector configuration", np.array([1.0,0.5,0.5]).reshape(3,1),0,0,0)
            # JointPosition("Joint 1 position", np.array([0]).reshape(1,1),0,0)
            
            
            #<<<<<<<<<<<<<Exercise-2>>>>>>>>>>>>>>>
            Position2D("End-effector position", np.array([1.0,0.5]).reshape(2,1), 0, np.array([1,1]),3),
            Orientation2D("End-effector orientation", np.array([0]).reshape(1,1), 0, np.array([1,1]),2)
        ] 


# Simulation params
dt = 1.0/60.0
count1 = -1

# Drawing preparation
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2,2))
ax.set_title('Simulation')
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x[m]')
ax.set_ylabel('y[m]')
line, = ax.plot([], [], 'o-', lw=2) # Robot structure
path, = ax.plot([], [], 'c-', lw=1) # End-effector path
point, = ax.plot([], [], 'rx') # Target

# stroing data
PPx = []
PPy = []
Time = []
err = [[] for _ in tasks]

# Simulation initialization
def init():
    global tasks, count1
    if tasks[0].name == "End-effector configuration":
        tasks[0].setDesired(np.array([np.random.uniform(-1.5,1.5), 
                                    np.random.uniform(-1.5,1.5), 0.2]).reshape(3,1))
    else:
        tasks[0].setDesired(np.array([np.random.uniform(-1.5,1.5), 
                            np.random.uniform(-1.5,1.5)]).reshape(2,1))
        # tasks[0].setFFVelocity(np.ones((2,1)))
        tasks[0].setK(np.diag([1,1]))
        
    count1 = count1 + 1                
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot, count, max_velocity
    global PPx, PPy, Time
    
    ### Recursive Task-Priority algorithm
    # Initialize null-space projector
    P = np.eye(robot.getDOF())
    # Initialize output vector (joint velocity)
    dq = np.zeros(robot.getDOF()).reshape(robot.getDOF(),1)
    count = 0
    # Loop over tasks
    for task in tasks:
        # Update task state
        task.update(robot)
        # Compute augmented Jacobian
        J_bar= task.getJacobian()@ P
        
        # <<<<<<<<<<<<<<Exercise 1>>>>>>>>>>>>>>>>>
        # Compute task velocity and Accumulate velocity
        # dq =dq + DLS(J_bar,0.1) @ (task.getError() - task.getJacobian() @ dq) 
        
        # <<<<<<<<<<<<<<Exercise 2>>>>>>>>>>>>>>>>>
        dq = dq + DLS(J_bar,0.1) @ (task.getFFVelocity() + task.getK() @ task.getError() - task.getJacobian() @ dq) 
        
        # #velocity scalling
        # s = np.max(dq/max_velocity)
        # if s > 1:
        #     dq = dq/s
        # Update null-space projector
        P = P - np.linalg.pinv(J_bar) @ J_bar
        
        # err[count].append(np.linalg.norm(task.getError))
        err[count].append(np.linalg.norm(task.getError()))
        count = count + 1

    ###

    # Update robot
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    Time.append(t + 10 * count1)
    path.set_data(PPx, PPy)
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

figure = plt.figure()


#<<<<<<<<<<<Exercise 1>>>>>>>>>>>>>>>>
# plt.plot(Time, err[0], label = 'End effector position task') #end-effector position task
# plt.plot(Time, err[1], label = 'End effector orientation task') #end-effector orientation task
# plt.plot(Time, err[0], label = 'End effector configuration task') #end-effector configuration task
# plt.plot(Time, err[1], label = 'Joint 1 position task') #end-joint 1 position task

#<<<<<<<<<<Exercise-2>>>>>>>>>>>>>>>>>>
plt.plot(Time, err[0], label = 'End effector position task') #end-effector position task
plt.plot(Time, err[1], label = 'Orientation task with 2nd link 0') #end-effector orientation task

plt.ylabel('Error[m]') #Title of the Y axis
plt.xlabel('Time[s]') #Title of the X axis
plt.title('Task-Priority')#Title of plot-1
plt.grid(True) #grid
plt.legend() #placing legend
plt.show()

