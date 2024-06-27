import numpy as np # Import Numpy

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
    

    # Rotation about Z-axis by theta
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                   [np.sin(theta), np.cos(theta), 0, 0],
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
                   [0, np.cos(alpha), -np.sin(alpha), 0],
                   [0, np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 0, 1]])

    # DH Transformation Matrix
    T = Rz @ Tz @ Tx @ Rx

    return T
    
def kinematics(d, theta, a, alpha, tb):
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
    T = [tb] # Base transformation
    Transformation = T
    # For each set of DH parameters:
    # 1. Compute the DH transformation matrix.
    # 2. Compute the resulting accumulated transformation from the base frame.
    # 3. Append the computed transformation to T.
    for i in range(len(d)):
        # Compute the DH transformation matrix for the current joint
        T_current = DH(d[i], theta[i], a[i], alpha[i])
        
        # Compute the accumulated transformation from the base
        T = T @ T_current
        
        # Append the accumulated transformation to the list
        Transformation = np.vstack((Transformation, np.array(T)))
        
    return Transformation

# Inverse kinematics
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

# Damped Least-Squares
def DLS(J, damping):
    '''
        Function computes the damped least-squares (DLS) solution to the matrix inverse problem.

        Arguments:
        A (Numpy array): matrix to be inverted
        damping (double): damping factor

        Returns:
        (Numpy array): inversion of the input matrix
    '''
    I = len(J)  # Identity matrix for a two-jointed robot
 
    damped_J = np.transpose(J) @ np.linalg.inv(J @ np.transpose(J) + ((damping ** 2) * np.identity(I))) 

   
    return damped_J# Implement the formula to compute the DLS of matrix A.

def Weighted_DLS(A, damping, weights):
    '''
    Function computes the damped least-squares (DLS) solution to the
    matrix inverse problem.
    Arguments:
    A (Numpy array): matrix to be inverted
    damping (double): damping factor
    weights (list): weight of each DOF of the robot
    Returns:
    (Numpy array): inversion of the input matrix
    '''
    l=len(A)
    W = np.diag(weights)
    return np.linalg.pinv(W) @ np.transpose(A) @ np.linalg.pinv(A @ np.linalg.pinv(W) @ np.transpose(A)+((damping ** 2)* np.identity(l)))

# Extract characteristic points of a robot projected on X-Y plane
def robotPoints2D(T):
    '''
        Function extracts the characteristic points of a kinematic chain on a 2D plane,
        based on the list of transformations that describe it.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
    
        Returns:
        (Numpy array): an array of 2D points
    '''
    P = np.zeros((2,len(T)))
    for i in range(len(T)):
        P[:,i] = T[i][0:2,3]
    return P