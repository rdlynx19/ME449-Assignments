import numpy as np
import modern_robotics as mr
import ur5_parameters as ur5


# To run the function for each case, pass the case number as argument to the run_sim function. run_sim will call the Puppet function to simulate the case. Minor changes might be required in run_sim depending upon the case you want to simulate. Please be careful of the Ftip value in Puppet function.


def run_sim(sim_case, thetalist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dthetalist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), restLength=0.0):
    """
    Run the simulation case based on numerical input.
    :param sim_case: The simulation case 1, 2, 3 or 4 as described in the assignment
    :param thetalist: an n-vector of initial joint angles (set to zero since we are starting from the home configuration)
    :param dthetalist: an n-vector of initial joint rates (set to zero since we are starting from rest)
    :param restLength: a scalar indicating the length of the spring when it is at rest (is set to zero for all simulation cases)
    """
    match sim_case:
        case 1:
            g = np.array([0.0, 0.0, -9.81])
            damping = 0.0
            stiffness = 0.0
            t = 5.0
            dt = 0.05 # change this value for (a) and (b)
            thetamat, dthetamat = Puppet(thetalist, dthetalist, g, ur5.Mlist, ur5.Slist, ur5.Glist, t, dt, damping, stiffness, restLength)

            np.savetxt("part1b.csv", thetamat, delimiter=',') # change filename for (a) and (b)

        case 2:
            g = np.array([0.0, 0.0, -9.81])
            damping = -0.005 # change this value of (a) and (b)
            stiffness = 0.0
            t = 5.0
            dt = 0.01
            thetamat, dthetamat = Puppet(thetalist, dthetalist, g, ur5.Mlist, ur5.Slist, ur5.Glist, t, dt, damping, stiffness, restLength)

            np.savetxt("part2b.csv", thetamat, delimiter=',')# change filename for (a) and (b)

        case 3:
            g = np.array([0.0, 0.0, 0.0])
            damping = 2.0 # change this value for (a) and (b)
            stiffness = 4.0
            t = 10.0
            dt = 0.01
            thetamat, dthetamat = Puppet(thetalist, dthetalist, g, ur5.Mlist, ur5.Slist, ur5.Glist, t, dt, damping, stiffness, restLength)

            if damping == 0.0:
                np.savetxt('part3a.csv', thetamat, delimiter=',')
            else:
                np.savetxt("part3b.csv", thetamat, delimiter=',')
        case 4:    
            g = np.array([0.0, 0.0, 0.0])
            damping = 2.0
            stiffness = 4.0
            t = 10.0
            dt = 0.01
            thetamat, dthetamat = Puppet(thetalist, dthetalist, g, ur5.Mlist, ur5.Slist, ur5.Glist, t, dt, damping, stiffness, restLength)

            np.savetxt("part4.csv", thetamat, delimiter=',') 

def Puppet(thetalist, dthetalist, g, Mlist, Slist, Glist, t, dt, damping, stiffness, restLength):
    """
    Make the robot behave like a puppet on a string.

    :param thetalist: an n-vector of initial joint angles (units: rad)
    :param dthetalist: an n-vector of initial joint rates (units; rad/s)
    :param g: the gravity vector in the {s} frame (units: m/s^2)
    :param Mlist: the configurations of the link frames relative to each other at the home configuration.
    :param Slist: the screw axes S_i in the space frame when the robot is at its home configuration
    :param Glist: the spatial inertia matrices G_i fo the links (units: kg and kg m^2)
    :param t: the total simulation time (untis: s)
    :param dt: the simulation timestep (units: s)
    :damping: a scalar indicating the viscous damping at each joint (units: Nms/rad)
    :stiffness: a scalar indicating the stiffness of the springy string (units: N/m)
    :restLength: a scalar indicating the length of the spring when it is at rest (units: m)

    :return thetamat: an Nxn matrix where row i is the set of joint values after simulation step i - 1
    :return dthetamat: an Nxn matrix where row i is the set of joint rates after simulation step i - 1 
    """
    # Lists to store the joint angles and velocities after each iteration
    thetamat = []
    dthetamat = []
    # Total iterations to run the simulation
    simIterations = (int)(t/dt)

    # The end effector configuration when it is at home position expressed in the {s} frame
    Msb = np.eye(4)
    for i in range(0,7):
        Msb = np.matmul(Msb , np.asanyarray(Mlist[i]))

    # freefall_taulist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Ftip = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    for i in range(0, simIterations):
        # Getting the current position of the spring
        springPos = referencePos(i*dt)
        # Calculating end effector position in {s} frame
        Tsb = mr.FKinSpace(Msb, Slist, thetalist)
        endEffPos = Tsb[0:3,-1]

        # Calculate extension of the spring
        spring_extension = np.linalg.norm(endEffPos - springPos) - restLength
        f_vecS = mr.Normalize(endEffPos - springPos) * (stiffness*spring_extension)
        F_vecS = np.array([0.0, 0.0, 0.0, f_vecS[0], f_vecS[1], f_vecS[2]])
        # Transforming the wrench into {b} frame
        F_vecB = np.matmul(mr.Adjoint(Tsb).T ,F_vecS)
        # comment out this line for cases 1 and 2
        Ftip = np.array([0.0, 0.0, 0.0, F_vecB[3], F_vecB[4], F_vecB[5]]) 
        
        # Torque values for damping, taulist is zero if damping is set to zero
        damped_taulist = - damping * dthetalist

        # Using ForwardDynamics to calculate joint accelerations
        joint_accel = mr.ForwardDynamics(thetalist, dthetalist, damped_taulist, g, Ftip, Mlist, Glist, Slist)
        # Calculating joint positions and velocities using Euler Integration(EulerStep)
        joint_angles, joint_velocities = mr.EulerStep(thetalist, dthetalist, joint_accel, dt)

        thetamat.append(joint_angles)
        dthetamat.append(joint_velocities)
        thetalist = joint_angles
        dthetalist = joint_velocities
    
    return np.asanyarray(thetamat), np.asanyarray(dthetamat)


def referencePos(t):
    """
    Return the current position of the spring.

    :param t: the current simulation time
    :return: The current spring position
    """
    # Constant spring position for Part 3
    # return np.array([0, 1, 1])

    # Sinusoidal spring position for Part4
    return np.array([1, np.cos(0.4*np.pi*t), 1])


run_sim(4) # enter case number to be simulated



