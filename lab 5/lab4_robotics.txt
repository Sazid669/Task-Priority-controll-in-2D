from lab2_robotics import * # Includes numpy import

def jacobianLink(T, revolute, link): # Needed in Exercise 2
    '''
        Function builds a Jacobian for the end-effector of a robot,
        described by a list of kinematic transformations and a list of joint types.

        Arguments:
        T (list of Numpy array): list of transformations along the kinematic chain of the robot (from the base frame)
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
        link(integer): index of the link for which the Jacobian is computed

        Returns:
        (Numpy array): end-effector Jacobian
    '''
    # Code almost identical to the one from lab2_robotics...
    # Number of joints up to the specified link
    n = len (T)-1
    
    # Initialize the Jacobian matrix
    J = np.zeros((6, n))
    
    # Position of the end-effector
    p_n = T[link][:3, 3]
    
    for i in range(link):
        # Extract the rotation matrix and position vector for the current joint
        R_i = T[i][:3, :3]
        p_i = T[i][:3, 3]
        
        # Compute the z-axis (rotation/translation axis) for the current joint
        z_i = R_i[:, 2]
        
        # Compute the vector from the current joint to the end-effector
        r = p_n - p_i
        
        if revolute[i]:
            # For revolute joints, compute the linear velocity component
            J[:3, i] = np.cross(z_i, r)
            # And the angular velocity component
            J[3:, i] = z_i
        else:
            # For prismatic joints, the linear velocity component is the z-axis
            J[:3, i] = z_i
            # And the angular velocity component is zero
            J[3:, i] = 0
    
    return J
    

'''
    Class representing a robotic manipulator.
'''
class Manipulator:
    '''
        Constructor.

        Arguments:
        d (Numpy array): list of displacements along Z-axis
        theta (Numpy array): list of rotations around Z-axis
        a (Numpy array): list of displacements along X-axis
        alpha (Numpy array): list of rotations around X-axis
        revolute (list of Bool): list of flags specifying if the corresponding joint is a revolute joint
    '''
    def __init__(self, d, theta, a, alpha, revolute):
        self.d = d
        self.theta = theta
        self.a = a
        self.alpha = alpha
        self.revolute = revolute
        self.dof = len(self.revolute)
        self.q = np.zeros(self.dof).reshape(-1, 1)
        self.update(0.0, 0.0)

    '''
        Method that updates the state of the robot.

        Arguments:
        dq (Numpy array): a column vector of joint velocities
        dt (double): sampling time
    '''
    def update(self, dq, dt):
        self.q += dq * dt
        for i in range(len(self.revolute)):
            if self.revolute[i]:
                self.theta[i] = self.q[i]
            else:
                self.d[i] = self.q[i]
        self.T = kinematics(self.d, self.theta, self.a, self.alpha)

    ''' 
        Method that returns the characteristic points of the robot.
    '''
    def drawing(self):
        return robotPoints2D(self.T)

    '''
        Method that returns the end-effector Jacobian.
    '''
    def getEEJacobian(self):
        return jacobian(self.T, self.revolute)

    '''
        Method that returns the end-effector transformation.
    '''
    def getEETransform(self):
        return self.T[-1]

    '''
        Method that returns the position of a selected joint.

        Argument:
        joint (integer): index of the joint

        Returns:
        (double): position of the joint
    '''
    def getJointPos(self, joint):
        return self.q[joint,0]

    '''
        Method that returns number of DOF of the manipulator.
    '''
    def getDOF(self):
        return self.dof
    
    def getLinkTransform(self, link):
        return self.T[link]

    '''
        Method that returns the link Jacobian.
    '''
    def getLinkJacobian(self, link):
        return jacobianLink(self.T, self.revolute, link)


'''
    Base class representing an abstract Task.
'''
class Task:
    '''
        Constructor.

        Arguments:
        name (string): title of the task
        desired (Numpy array): desired sigma (goal)
    '''
    def __init__(self, name, desired, FFVelocity= None, K=None):
        self.name = name # task title
        self.sigma_d = desired # desired sigma
        self.FFVelocity = FFVelocity #feedforward velocity
        self.K = K #gain matrix
        
    '''
        Method updating the task variables (abstract).

        Arguments:
        robot (object of class Manipulator): reference to the manipulator
    '''
    def update(self, robot):
        pass

    ''' 
        Method setting the desired sigma.

        Arguments:
        value(Numpy array): value of the desired sigma (goal)
    '''
    def setDesired(self, value):
        self.sigma_d = value

    '''
        Method returning the desired sigma.
    '''
    def getDesired(self):
        return self.sigma_d

    '''
        Method returning the task Jacobian.
    '''
    def getJacobian(self):
        return self.J

    '''
        Method returning the task error (tilde sigma).
    '''    
    def getError(self):
        return self.err
    
    def setFFVelocity(self, value):
        self.FFVelocity = value

    '''
        Method returning the feedforward velocity vector.
    '''
    def getFFVelocity(self):
        return self.FFVelocity

    ''' 
        Method setting the gain matrix K.

        Arguments:
        value(Numpy array): value of the gain matrix K.
    '''
    def setK(self, value):
        self.K = value

    '''
        Method returning the gain matrix K.
    '''
    def getK(self):
        return self.K
    



'''
    Subclass of Task, representing the 2D position task.
'''
class Position2D(Task):
    def __init__(self, name, desired, FFVelocity=None, K= None, link= None):
        super().__init__(name, desired, FFVelocity, K)
        self.J = np.zeros((2,3)) # Initializing with proper dimensions
        self.err = np.zeros((2,1)) # Initializing with proper dimensions
        self.FFVelocity = np.zeros((2,1)) # Initializing with proper dimensions
        self.K = np.eye((2)) # Initializing with proper dimensions
        self.link = link
        
    def update(self, robot):
        #<<<<<<<<Exercise-1>>>>>>>>>>
        self.J=robot.getEEJacobian()[:2,:] 
        sigma = robot.getEETransform()[:2,3].reshape(2,1)  # Current position of the task
        self.err =  self.getDesired() - sigma #task error

    def isActive(self):
        return 1 
'''
    Subclass of Task, representing the 2D orientation task.
'''
class Orientation2D(Task):
    def __init__(self, name, desired, FFVelocity, K, link):
        super().__init__(name, desired, FFVelocity, K)
        self.J = np.zeros((1,3)) # Initialize with proper dimensions
        self.err = np.zeros((1,1)) # Initialize with proper dimensions
        self.FFVelocity = np.zeros((1,1)) # Initialize with proper dimensions
        self.K = np.eye((1)) # Initialize with proper dimensions
        self.link = link

    def update(self, robot):
        #<<<<<<<<Exercise-1>>>>>>>>>>>
        self.J = robot.getEEJacobian()[5,:].reshape(1,3)
        angle = np.arctan2(robot.getEETransform()[1,0], robot.getEETransform()[0,0])
        self.err = (self.getDesired() - angle)

        
'''
    Subclass of Task, representing the 2D configuration task.
'''
class Configuration2D(Task):
    def __init__(self, name, desired, FFVelocity, K, link):
        super().__init__(name, desired, FFVelocity, K)
        self.J = np.zeros((3,3)) # Initializing with proper dimensions
        self.err = np.zeros((3,1)) # Initializing with proper dimensions
        self.FFVelocity = np.zeros((3,1)) # Initializing with proper dimensions
        self.K = np.eye((3)) # Initializing with proper dimensions
        self.link = link

    def update(self, robot):
        #<<<<<<<Exercise-1>>>>>>>>
        self.J[:2,:] = robot.getEEJacobian()[:2,:] 
        self.J[2,:] = robot.getEEJacobian()[5,:]
        angle = np.arctan2(robot.getEETransform()[1,0],robot.getEETransform()[0,0])
        self.err[:2]= self.getDesired()[:2] - robot.getEETransform()[:2,3].reshape(2,1)
        self.err[2] = self.getDesired()[2] - angle

''' 
    Subclass of Task, representing the joint position task.
'''
class JointPosition(Task):
    def __init__(self, name, desired, FFVelocity, K):
        super().__init__(name, desired, FFVelocity, K)
        self.J = np.zeros((1,3)) # Initializing with proper dimensions
        self.err = np.zeros((1,1)) # Initializing with proper dimensions
        self.FFVelocity = np.zeros((1,1)) # Initializing with proper dimensions
        self.K = np.eye((1)) # Initializing with proper dimensions

    def update(self, robot):
        self.J[0,0] = 1 #for joint 1
        self.err =  self.getDesired() - robot.getJointPos(0)


''' 
    Subclass of Task, representing 2D circular obostracle.
'''
class Obstacle2D(Task):
    def __init__(self, name, obstacle_pos, radius):
        super().__init__(name, 0)
        self.obstacle_pos = obstacle_pos
        self.radius = radius
        self.J = np.zeros((2,3)) # Initializing with proper dimensions
        self.err = np.zeros((2,1)) # Initializing with proper dimensions
        self.active = 0
        self.distance = 0
    
    def isActive(self):
        return self.active   
    
    def update(self, robot):
        self.J = robot.getEEJacobian()[:2, :]
        current_pos = robot.getEETransform()[:2,3].reshape(2,1)
        self.err =  (current_pos - self.obstacle_pos)/np.linalg.norm(current_pos - self.obstacle_pos)
        self.distance = current_pos - self.obstacle_pos
        if self.active==0 and np.linalg.norm(current_pos - self.obstacle_pos) <= self.radius[0]:
            self.active=1
        elif self.active==1 and np.linalg.norm(current_pos - self.obstacle_pos) >= self.radius[1]:
            self.active=0

''' 
    Subclass of Task, representing the joint limit task.
'''
class JointLimit(Task):
    def __init__(self, name, safe_set, thresholds):
        super().__init__(name, 0)
        self.J = np.zeros((1,3)) # Initializing with proper dimensions
        self.err = np.zeros((1,1)) # Initializing with proper dimensions
        self.safe_set = safe_set
        self.thresholds = thresholds
        self.active = 0

    def update(self, robot):

        self.J[0,0] = 1 #for joint 1   
        self.err =  1 * self.active

        #Activation update
        if self.active==0 and robot.getJointPos(0) >= self.safe_set[1] - self.thresholds[0]:
            self.active = -1
        elif self.active== 0 and robot.getJointPos(0) <= self.safe_set[0] + self.thresholds[0]:
            self.active = 1
        elif self.active== -1 and robot.getJointPos(0) <= self.safe_set[1] - self.thresholds[1]:
            self.active = 0
        elif self.active==1 and robot.getJointPos(0) >= self.safe_set[0] + self.thresholds[1]:
            self.active = 0


    def isActive(self):
        return self.active 