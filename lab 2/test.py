import numpy as np

def DH(d, theta, a, alpha):
    '''
        Function builds elementary Denavit-Hartenberg transformation matrices 
        and returns the transformation matrix resulting from their multiplication.

        Arguments:
        d (double): displacement along Z-axis
        theta (double): rotation around Z-axis
        a (double): displacement along X-axis
        alpha (double): rotation around X-axis

        Returns:
        (Numpy array): composition of elementary DH transformations
    '''
    # 1. Build matrices representing elementary transformations (based on input parameters).
    # 2. Multiply matrices in the correct order (result in T).
    # Convert angles from degrees to radians for consistency
    theta_rad = np.radians(theta)
    alpha_rad = np.radians(alpha)

    # Rotation about Z-axis by theta
    Rz = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0, 0],
                   [np.sin(theta_rad), np.cos(theta_rad), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Translation along Z-axis by d
    Tz = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, d],
                   [0, 0, 0, 1]])

    # Translation along X-axis by a
    Tx = np.array([[1, 0, 0, a],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Rotation about X-axis by alpha
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(alpha_rad), -np.sin(alpha_rad), 0],
                   [0, np.sin(alpha_rad), np.cos(alpha_rad), 0],
                   [0, 0, 0, 1]])

    # DH Transformation Matrix
    T = Rz @ Tz @ Tx @ Rx

    return T

def kinematics(d, theta, a, alpha):
    '''
        Functions builds a list of transformation matrices, for a kinematic chain,
        descried by a given set of Denavit-Hartenberg parameters. 
        All transformations are computed from the base frame.

        Arguments:
        d (list of double): list of displacements along Z-axis
        theta (list of double): list of rotations around Z-axis
        a (list of double): list of displacements along X-axis
        alpha (list of double): list of rotations around X-axis

        Returns:
        (list of Numpy array): list of transformations along the kinematic chain (from the base frame)
    '''
    T = [np.eye(4)] # Base transformation
    # For each set of DH parameters:
    # 1. Compute the DH transformation matrix.
    # 2. Compute the resulting accumulated transformation from the base frame.
    # 3. Append the computed transformation to T.
    

    for i in range(0,len(d)):
        # Compute the DH transformation matrix for the current joint
        T_current = DH(d[i], theta[i], a[i], alpha[i])
        # Compute the accumulated transformation from the base
        T_accumulated = T[-1] @ T_current
        
        # Append the accumulated transformation to the list
        T.append(T_accumulated)
        
        
    return T


def jacobian(T, revolute):
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # 1. Initialize J and O.
    # 2. For each joint of the robot
    #   a. Extract z and o.
    #   b. Check joint type.
    #   c. Modify corresponding column of J.
    n = len(T)-1  # Number of joints/frames
    J = np.zeros((6, n))  # Initialize the Jacobian matrix with zeros
    
    O = np.array([T[-1][:3, 3]]).T  # End-effector's origin (position)
    Z = np.array([[0, 0, 1]]).T  # Z-axis of the base frame
    
    for i in range(n):
        # Extract the rotation matrix and origin from the transformation matrix
        R_i = T[i][:3, :3]
        O_i = np.array([T[i][:3, 3]]).T
        
        # Extract the z-axis from the rotation matrix
        Z_i = R_i @ Z
        
        if revolute[i]:
            # For revolute joints, use the cross product of z and (O - O_i)
            J[:3, i] = np.cross(Z_i.T, (O - O_i).T).T[:, 0]
            J[3:, i] = Z_i[:, 0]
        else:
            # For prismatic joints, the linear velocity is along the z-axis, and angular velocity is zero
            J[:3, i] = Z_i[:, 0]
            # The angular part is zero for prismatic joints, which is already set by the initialization with zeros
    
    return J

d = np.zeros(2)           # displacement along Z-axis
q = np.array([0.2, 0.5])  # rotation around Z-axis (theta)
a = np.array([0.75, 0.5]) # displacement along X-axis
alpha = np.zeros(2)  
revolute = [True, True]
Y=kinematics(d, q, a, alpha)
J=jacobian(Y,revolute)
print(J)
J_2d = J[:2, :]  # Extract the first two rows for the x and y translational movements
J_rot = J[5:6, :]  # Extract the sixth row for rotational movement
Jocobian = np.vstack((J_2d, J_rot))
print(Jocobian)
print(J)


