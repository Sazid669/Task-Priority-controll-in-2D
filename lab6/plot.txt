import matplotlib.pyplot as plt #Plotting library
import numpy as np # linear algebra library 


# Define three arrays loaded from disk files
F_R=np.load('move forward then rotate.npy',allow_pickle=True)
R_F=np.load(' rotate then move forward.npy',allow_pickle=True)
F_and_R=np.load(' rotate and move forward.npy',allow_pickle=True)

# Create a new figure object
fig = plt.figure()
# Add subplot to the current figure on a "1x1 grid, with ID: first subplot", with autoscale.
ax = fig.add_subplot(111, autoscale_on=True)
# Title of the plot
ax.set_title('Mobile Manipulator position on the X-Y plane')
# Label of x axis
ax.set_xlabel('x [m]')
# Label of y axis
ax.set_ylabel('y [m]')
# Aspect of axes (the ratio of y-unit to x)
ax.set_aspect('auto')
# Grid of the subplot
ax.grid()
# Plot presenting the evolution of the mobile base position and the end-effector position on the X-Y plane
ax.plot(F_R[0][0], F_R[0][1], marker="x", color='purple', label='end-effector position F>R')
ax.plot(F_R[1][0], F_R[1][1], marker=".", color='cyan', label='base position F>R')
ax.plot(R_F[0][0], R_F[0][1], marker="x", color='blue', label='end-effector position R>F')
ax.plot(R_F[1][0], R_F[1][1], marker=".", color='black', label='base position R>F')
ax.plot(F_and_R[0][0], F_and_R[0][1], marker="x", color='yellow', label='end-effector position F&R')
ax.plot(F_and_R[1][0], F_and_R[1][1], marker=".", color='salmon', label='base position F&R')
# Legend
ax.legend()
# Display the plot
plt.show()