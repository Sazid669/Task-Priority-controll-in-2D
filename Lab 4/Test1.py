from lab4_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot model - 3-link manipulator
d = np.zeros(3)                            # displacement along Z-axis
theta = np.array([0.2, 0.5, 0.1])     # rotation around Z-axis
alpha = np.zeros(3)                        # rotation around X-axis
a =  np.array([0.75, 0.5, 0.45])                # displacement along X-axis
revolute =  [True, True, True]                    # flags specifying the type of joints
robot = Manipulator(d, theta, a, alpha, revolute) # Manipulator object

# Task hierarchy definition
tasks = [  
            #<<<======for exercise-1=======>
            #Position2D("End-effector position", np.array([1.0,0.5]).reshape(2,1),0,0,0),
            #Orientation2D("End-effector orientation", np.array([0]).reshape(1,1),0,0,0)
            #Configuration2D("End-effector configuration", np.array([1.0,0.5,0.2]).reshape(3,1),0,0,0)
            #JointPosition("Joint 1 position", np.array([0]).reshape(1,1),0,0)

            #<<<======for excercise-2=======>
            Position2D("End-effector position", np.array([1.0,0.5]).reshape(2,1), 0, np.array([1,1]),3),
            Orientation2D("End-effector orientation", np.array([0]).reshape(1,1), 0, np.array([1,1]),2)
        ] 

# Simulation params
dt = 1.0/60.0
counter = -1  #for counting simulation loop

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
PPx = []
PPy = []
time= []
error = [[],[]]
# Simulation initialization
def init():
    global tasks, counter
    if tasks[0].name =="End-effector configuration":
        tasks[0].setDesired(np.array([np.random.uniform(-1.5,1.5), 
                            np.random.uniform(-1.5,1.5), 0.2]).reshape(3,1))

    else: #when task[0] is End-effector position
        tasks[0].setDesired(np.array([np.random.uniform(-1.5,1.5), 
                            np.random.uniform(-1.5,1.5)]).reshape(2,1))
        #for exercise 2                    
        tasks[0].setK(np.diag([5,5]))
        #####tasks[0].setFFVelocity(np.ones((2,1)))

    counter=counter+1

    line.set_data([], [])
    path.set_data([], [])
    point.set_data([], [])
    return line, path, point

# Simulation loop
def simulate(t):
    global tasks
    global robot
    global PPx, PPy

    ### Recursive Task-Priority algorithm
    # Initialize null-space projector
    P = np.eye(robot.getDOF())
    # Initialize output vector (joint velocity)
    dq = np.zeros(robot.getDOF()).reshape(robot.getDOF(),1)

    count=0
    for i in tasks: # Looping over tasks
        i.update(robot) # Updating task state
        J_bar= i.getJacobian()@ P # Computing augmented Jacobian

        #<<<<<====== Part of Exercise-1 ======>>>>>
        # Computing and accumulating task velocity
        #dq =dq + DLS(J_bar,0.1) @ (i.getError() - i.getJacobian() @ dq) 
       
        #<<<<<====== Part of Exercise-2 ======>>>>>
        # Computing and accumulating task velocity
        dq = dq + DLS(J_bar,0.1) @ (i.getFFVelocity() + i.getK() @ i.getError() - i.getJacobian() @ dq) 
        
        P = P - np.linalg.pinv(J_bar) @ J_bar  # Updating null-space projector
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
    point.set_data(tasks[0].getDesired()[0], tasks[0].getDesired()[1])

    return line, path, point

# Run simulation
animation = anim.FuncAnimation(fig, simulate, np.arange(0, 10, dt), 
                                interval=10, blit=True, init_func=init, repeat=True)
plt.show()

#plot-2
fig2 = plt.figure(2)

if(len(tasks)>1):
    if tasks[1].name == "End-effector orientation":
        #plt.plot(time, error[0], label='e1 (End-effector position)') #plotting position of error 1 against time
        #plt.plot(time, error[1], label='e2 (End-effector orientation)') #plotting position of error 2 against time
        #for exercise-2
        plt.plot(time, error[0], label='e1 (End-effector position), K= np.diag([5,5])') #plotting position of error 1 against time
        plt.plot(time, error[1], label='e2 (link 2 orientation)') #plotting position of error 2 against time
    else:
        plt.plot(time, error[0], label='e1 (End-effector position)') #plotting position of error 1 against time
        plt.plot(time, error[1], label='e2 (Joint 1 position)') #plotting position of error 2 against time
else:
    if tasks[0].name == "End-effector position":
        plt.plot(time, error[0], label='e1 (End-effector position)') #plotting position of error 1 against time
    else:
        plt.plot(time, error[0], label='e1 (End-effector configuration)') #plotting position of error 1 against time

plt.ylabel('Error[m]') #Title of the Y axis
plt.xlabel('Time[s]') #Title of the X axis
plt.title('Task-Priority')#Title of plot-1
plt.xlim(left=0) #setting x axis limit
plt.grid(True) #grid 
plt.legend() #placing legend
plt.show()