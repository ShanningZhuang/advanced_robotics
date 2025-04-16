# -*- coding: utf-8 -*-
# @Author : zhux
# @Email : zhuxiang24@mails.tsinghua.edu.cn

import os, time, copy, math
import numpy as np
import pybullet_data
import pybullet as p
from core import *
from robot_env import *


# for pick and place demo, use PICK_PLACE
TESTING = "IK"
# TESTING = "PICK_PLACE"


def IK_test(robot, physicsClientId):
    screw_axis = robot.get_screw_axis()
    M = robot.get_end_eff_home_trans()

    R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((3.1415926, 0, 0)))).reshape(3, 3)
    P = np.array([0.3, 0.3, 0.4])

    theta_0 = np.array([0.0] * 7)

    def gradual_IK(thetalist0, P, R):
        T = rp_to_trans(R, P)
        thetalist, state = InverseKinematics_in_space(np.array(screw_axis).T, M, T, thetalist0, 0.001, 0.0005)

        act = thetalist

        robot.apply_action(act, max_vel=1)
        for _ in range(240):
            p.stepSimulation(physicsClientId=physicsClientId)
            time.sleep(1 / 240)
        M_new = robot.cal_eff_trans()
        return M_new, act

    print("---IK testing---")
    M_new, last_act = gradual_IK(theta_0, P, R)
    print("---Robot eff pose---")
    print(M_new)
    M_old = rp_to_trans(R, P)
    print(np.linalg.norm(M_new-M_old))
    print("\nyou should find that M_new is the same as your desired pose")
    print("---finish IK test---")


def pick_and_place_demo(robot, physicsClientId):
    robot.load_env()
    p.setGravity(0, 0, -9.8, physicsClientId=physicsClientId)
    action_ll, action_ul, _, _ = robot.get_joint_action_ranges()
    joint_limits = []
    for i in range(7):
        joint_limits.append((action_ll[i], action_ul[i]))
    robot.debug_gui()
    robot.pre_grasp()
    screw_axis = robot.get_screw_axis()
    M = robot.get_end_eff_home_trans()

    # eff orientation
    R = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((3.1415926, 0, 0)))).reshape(3, 3)
    # initial theta
    thetalist0 = [0] * 7

    # define the via points
    via_point_pick = [
        # TODO:  --- Your code start ---

        # TODO:  --- Your code ends ---
    ]
    via_point_place = [
        # TODO:  --- Your code start ---

        # TODO:  --- Your code ends ---
    ]
    # back your robot to the start position
    via_point_back = [
        # TODO:  --- Your code start ---

        # TODO:  --- Your code ends ---
    ]
    # Move the robot though via_point_pick points
    # TODO:  --- Your code start ---

    # TODO:  --- Your code ends ---

    robot.pre_grasp()
    step_sim(0.5, physicsClientId=physicsClientId)
    print("---Grasping!---")
    robot.grasp()
    step_sim(0.5, physicsClientId=physicsClientId)

    # Move the robot though via_point_place points
    # TODO:  --- Your code start ---

    # TODO:  --- Your code ends ---

    print("---Placing!---")
    robot.pre_grasp()
    step_sim(0.5, physicsClientId=physicsClientId)

    print("---Back to home!---")
    # Move the robot though via_point_back points
    # TODO:  --- Your code start ---

    # TODO:  --- Your code ends ---

    print("---Finish pick_and_place_demo---")


def test_panda(physicsClientId, controlMode="position"):
    # create robot
    robot = pandaEnv(physicsClientId, controlMode=controlMode)
    p.setGravity(0, 0, -9.8, physicsClientId=physicsClientId)

    robot.debug_gui()
    robot.pre_grasp()

    for _ in range(120):
        p.stepSimulation(physicsClientId=physicsClientId)
        time.sleep(1 / 240)
    if TESTING == "IK":
        IK_test(robot, physicsClientId)
    elif TESTING == "PICK_PLACE":
        pick_and_place_demo(robot, physicsClientId)

    while (1):
        p.stepSimulation(physicsClientId)
        time.sleep(1 / 240)


if __name__ == '__main__':
    physicsClientId = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=physicsClientId)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=physicsClientId)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=physicsClientId)
    test_panda(physicsClientId, controlMode="position")
