
# Import necessary libraries
from lab2_robotics import * # Includes numpy import
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Robot definition
d = np.zeros(2)           # displacement along Z-axis
q1 = np.array([0.2, 0.5]) # rotation around Z-axis (theta)[used in Transpose]
q2 = np.array([0.2, 0.5]) # rotation around Z-axis (theta) [used in Pseudoinverse]
q3 = np.array([0.2, 0.5]) # rotation around Z-axis (theta) [used in DLS]
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)       # rotation around X-axis 
revolute = [True, True]
sigma_d = np.array([0.0, 1.0])
K = np.diag([1, 1])

# Simulation params
dt = 1.0/60
damping=0.1




# Drawing preparation


fig3, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
titles = ['Transpose Method', 'Pseudoinverse Method', 'DLS Method']


for ax, title in zip(axs, titles):
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_title(title)  # Set the title for each subplot
    ax.set_xlabel('X Axis')  # Set the X-axis label for each subplot
    ax.set_ylabel('Y Axis')  # Set the Y-axis label for each subplot

# Initialize lines, paths, points for each subplot
lines = [ax.plot([], [], 'o-', lw=2)[0] for ax in axs]
paths = [ax.plot([], [], 'c-', lw=1)[0] for ax in axs]
points = [ax.plot([], [], 'rx')[0] for ax in axs]
PPx= [[], [], []]
PPy = [[], [], []]


dq_transpose_history = []
dq_dls_history = []
dq_pseudo_inverse_history = []
time_steps = []

# Simulation initialization function
def init():
    for line, path, point in zip(lines, paths, points):
        line.set_data([], [])
        path.set_data([], [])
        point.set_data([], [])
    return lines + paths + points

# Simulation loop
def simulate(t):
    global d, q1, q2, q3, a, alpha, revolute, sigma_d, K, dt
    global PPx, PPy, dq_dls_history, dq_pseudo_inverse_history, dq_transpose_history, time_steps, damping
    qs = [q1, q2, q3]  # List of joint values for each solution

    for i, q in enumerate(qs):
        print(i)
        print(q)
       
         # Update robot
        T = kinematics(d, q, a, alpha) #getting list of transformations
        J = jacobian(T, revolute)[:2,:] ##calculating jacobian and extracting 2x2 matrix 
        
        # Update control
        sigma = T[-1][:2, 3]  # X,Y Position of the end-effector
        err = sigma_d - sigma # Control error

        # Calculate each control solution  
        if i == 0:  # Transpose
            dq = J.T @ K @ err
         
            dq_transpose_history.append(np.linalg.norm(err))
        elif i == 1:  # Pseudoinverse
            dq = np.linalg.pinv(J) @ K @ err
            dq_dls_history.append(np.linalg.norm(err))
        else:  # DLS
            dq = DLS(J, damping) @ K @ err
            dq_pseudo_inverse_history.append(np.linalg.norm(err))
            

        # Update joint values
        qs[i] += dt * dq

        # Update drawing for each method
        P = robotPoints2D(T)
        lines[i].set_data(P[0, :], P[1, :])
        PPx[i].append(P[0, -1])
        PPy[i].append(P[1, -1])
        paths[i].set_data(PPx[i], PPy[i])
        points[i].set_data(sigma_d[0], sigma_d[1])
        
    time_steps.append(t)

    return lines + paths + points

# Create the animation
animation = anim.FuncAnimation(fig3, simulate, frames=np.arange(0, 10, dt),
                               init_func=init, blit=True, repeat=False)

plt.show()


#Plotting Evolution of error values over time
fig4=plt.figure(4)
plt.plot(time_steps, dq_transpose_history, label='Transpose')
plt.plot(time_steps, dq_dls_history, label='DLS')
plt.plot(time_steps,  dq_pseudo_inverse_history, label='PseudoInverse')

plt.xlabel('Time[s]')
plt.ylabel('Error[m]')
plt.title('Resolved Rate Motion Control')

plt.legend()
plt.grid(True)
plt.show()

