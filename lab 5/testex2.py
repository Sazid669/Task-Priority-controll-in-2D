from test import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patch

# Robot model
d = np.zeros(3)                     # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.1])   # rotation around Z-axis
alpha = np.zeros(3)                 # rotation around X-axis
a = np.array([0.75, 0.5, 0.45])     # displacement along X-axis
revolute = [True, True, True]       # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

# Task hierarchy definition
thresholds = np.array([0.03, 0.05]) #activation and deactivation thresholds
safe_set = np.array([-0.5, 0.5]) #safe_set: q_min,q_max


tasks = [ 
          JointLimit("Joint limits", safe_set, thresholds),
          Position2D("End-effector position", np.array([0.25, -0.75]).reshape(2,1))
        ] 

# Simulation params
dt = 1.0/60.0
counter = -1

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
#memory
PPx = []
PPy = []
time= []
error = [[],[]]

# Simulation initialization
def init():
    global tasks,counter

    counter = counter+1
    #end-effector position as final task
    tasks[len(tasks)-1].setDesired(np.array([np.random.uniform(-1.5,1.5), 
                            np.random.uniform(-1.5,1.5)]).reshape(2,1))
    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy
    
    ### Recursive Task-Priority algorithm (w/set-based tasks)
    # The algorithm works in the same way as in Lab4. 
    # The only difference is that it checks if a task is active.
    ###
    P = np.eye(robot.getDOF())
    # Initialize output vector (joint velocity)
    dq = np.zeros(robot.getDOF()).reshape(robot.getDOF(),1)

    count=0
    for i in tasks: # Looping over tasks
        i.update(robot) # Updating task state
        if i.IsActive():
            J_bar= i.getJacobian()@ P # Computing augmented Jacobian
            # Computing and accumulating task velocity
            dq =dq + DLS(J_bar,0.1) @ (i.getError() - i.getJacobian() @ dq) 
            P = P - np.linalg.pinv(J_bar) @ J_bar  # Updating null-space projector
        #appending joint position 
        if count == 0:
            error[count].append(robot.getJointPos(0))   
        else: #appending end-effector position error
            error[count].append(np.linalg.norm(i.getError()))

        count = count + 1

    # Update robot
    
    robot.update(dq, dt)
    
    # Update drawing
    PP = robot.drawing()
    line.set_data(PP[0,:], PP[1,:])
    PPx.append(PP[0,-1])
    PPy.append(PP[1,-1])
    time.append(t + 10 * counter)
    path.set_data(PPx, PPy)
    point.set_data(tasks[len(tasks)-1].getDesired()[0], tasks[len(tasks)-1].getDesired()[1])
    
    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

#plot-2
fig2 = plt.figure(2)
# specifying horizontal line for safe sets
plt.axhline(y = -0.5, color = 'r', linestyle = '--')
plt.axhline(y = 0.5, color = 'r', linestyle = '--')
plt.plot(time, error[0], label='q1 (position of joint 1)') #plotting position of joint 1 against time
plt.plot(time, error[1], label='e2 (end-effector position error)') #plotting position error of end-effector
plt.ylabel('Error[m]') #Title of the Y axis
plt.xlabel('Time[s]') #Title of the X axis
plt.title('Task-Priority control')#Title of plot-1
plt.xlim(left=0) #setting x axis limit
plt.grid(True) #grid 
plt.legend() #placing legend
plt.show()