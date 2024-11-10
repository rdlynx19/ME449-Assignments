import modern_robotics as mr
import numpy as np
import math
import csv
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt


W1 = 0.109
W2 = 0.082
L1 = 0.425
L2 = 0.392
H1 = 0.089
H2 = 0.095

Blist = np.array([[0, 1, 0, W1+W2, 0, L1+L2],
                  [0, 0, 1, H2, -L1 - L2, 0],
                  [0, 0, 1, H2, -L2, 0],
                  [0, 0, 1, H2, 0, 0],
                  [0, -1, 0, -W2, 0, 0],
                  [0, 0, 1, 0, 0, 0]]).T

M = np.array([[-1, 0, 0, L1+L2], 
             [0, 0, 1, W1+W2],
              [0, 1, 0, H1-H2],
              [0, 0, 0, 1]])

T = np.array([[1, 0, 0, 0.3],
             [0, 1, 0, 0.3],
             [0, 0, 1, 0.4],
             [0, 0, 0, 1]])
short_itr = np.array([0.76, -1.32,  2.44, 0.72,  -1.59, 1.4])

long_itr = np.array([np.pi/4, 0.0, -np.pi, -np.pi/4, np.pi/2, 0.05])

eomg = 0.001
ev = 0.0001

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return joint_angle_array: Array of joint angles for each iteration
    :return end_eff_pos: Array of the end effector position after each iteration.
    :return ang_err_list: List of angular error values for each iteration.
    :return lin_err_list: List of linear error values for each iteration.
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    joint_angle_list = []
    end_eff_pos = []
    ang_err_list = []
    lin_err_list = []
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, \
                                                      thetalist)), T)))
    err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
          or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev
    while err and i < maxiterations:
        

        print(f"\nIteration {i}")
        print("\njoint_vector: ")

        # Change the theta values to lie between -pi and pi
        for index, theta in np.ndenumerate(thetalist):
            theta_new = math.atan2(np.sin(theta), np.cos(theta))
            thetalist[index] = theta_new

        print(thetalist)

        joint_angle_list.append(thetalist)
        i = i+1

        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, \
                                                         thetalist)), Vb)

        # Get the end effector pose for the current joint angles
        end_eff_config = mr.FKinBody(M, Blist, thetalist)
        end_eff_pos.append(end_eff_config[0:3,-1])
        print("\n SE(3) end-effector config: ")
        print(end_eff_config)

        Vb \
        = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(mr.FKinBody(M, Blist, \
                                                       thetalist)), T)))
        
        print(f"\nerror twist V_b: {Vb}")
        ang_err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) 
        lin_err = np.linalg.norm([Vb[3], Vb[4], Vb[5]])
        ang_err_list.append(ang_err)
        lin_err_list.append(lin_err)
        print(f"\nangular error ||omega_b||: {ang_err}")
        print(f"\nlinear_error ||v_b||: {lin_err}")

        err = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) > eomg \
              or np.linalg.norm([Vb[3], Vb[4], Vb[5]]) > ev

 
    joint_angle_array = np.asanyarray(joint_angle_list)
    print(f"Joint Angle Matrix: {joint_angle_array} ")  

    return (thetalist, joint_angle_array, end_eff_pos, ang_err_list, lin_err_list,not err)


short_itr_x = []
short_itr_y = []
short_itr_z = []


def short_itr_call(short_itr):
    """Calling the IKinBodyIterates function for the 'good' initial guess."""
    joint_vector, joint_angle_array, end_eff_pos, sh_ang_errs, sh_lin_errs, success = IKinBodyIterates(Blist, M, T, short_itr, eomg, ev)

    joint_angle_array_rounded = np.round(joint_angle_array, decimals=5)

    np.savetxt("short_itr.csv", joint_angle_array_rounded, delimiter= ",")

    for arr in end_eff_pos:
        short_itr_x.append(arr[0])
        short_itr_y.append(arr[1])
        short_itr_z.append(arr[2])
    return sh_ang_errs, sh_lin_errs
    
    
long_itr_x = []
long_itr_y = []
long_itr_z = []


def long_itr_call(long_itr):
    """Calling the IKinBodyIterates function for the 'bad(long convergence)' initial guess."""
    joint_vector, joint_angle_array, end_eff_pos, ln_ang_err, ln_lin_err, success = IKinBodyIterates(Blist, M, T, long_itr, eomg, ev)

    joint_angle_array_rounded = np.round(joint_angle_array, decimals=5)

    np.savetxt("long_itr.csv", joint_angle_array_rounded, delimiter= ",")

    for arr in end_eff_pos:
        long_itr_x.append(arr[0])
        long_itr_y.append(arr[1])
        long_itr_z.append(arr[2])
    return ln_ang_err, ln_lin_err

sh_ang_errs, sh_lin_errs = short_itr_call(short_itr)
ln_ang_err, ln_lin_err = long_itr_call(long_itr)

fig1 = plt.figure()
ax1 = plt.subplot(projection='3d')
ax1.set_title('ee poses from start to end')

ax1.plot(short_itr_x, short_itr_y, short_itr_z,  linestyle='-', color='blue', label='short')
ax1.plot(long_itr_x, long_itr_y, long_itr_z, linestyle='-', color='purple', label='long')
ax1.scatter(short_itr_x[0], short_itr_y[0], short_itr_z[0], marker='o', color='orange')
ax1.scatter(long_itr_x[0], long_itr_y[0], long_itr_z[0], marker='o', color='orange', label='start')
ax1.scatter(short_itr_x[2], short_itr_y[2], short_itr_z[2], marker='x', color='black', label='end')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.legend()
plt.show()


fig2, ax2 = plt.subplots(1,1)
ax2.set_title('Angular error v/s iteration')

itr_list_long = np.linspace(0, 16, 16)
itr_list_short = np.linspace(0, 3, 3)

ax2.plot(itr_list_short, sh_ang_errs, label='short') 
ax2.plot(itr_list_long, ln_ang_err, label='long')

ax2.set_xlabel('Iteration Number')
ax2.set_ylabel('AngularError')
plt.legend()
plt.show()