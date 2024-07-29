# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:08:14 2020

@author: mrobinson2
"""

import re

delsys_dtypes = {'L1 Erector Spinae LEFT' : float,
                 'L1 Erector Spinae RIGHT': float,
                 'L3 Erector Spinae LEFT' : float,
                 'L3 Erector Spinae RIGHT': float,
                 'L5 Multifidus LEFT'     : float,
                 'L5 Multifidus RIGHT'    : float,
                 'Rectus Abdominis RIGHT' : float,
                 'Rectus Abdominis LEFT'  : float,
                 'External Oblique RIGHT' : float,
                 'External Oblique LEFT'  : float,
                 'Gluteus Medius LEFT'    : float,
                 'Gluteus Medius RIGHT'   : float,
                 'Vastus Lateralis RIGHT' : float,
                 'Vastus Lateralis LEFT'  : float,
                 'Biceps Femoris LEFT'    : float,
                 'Biceps Femoris RIGHT'   : float,
                 'time'                   : float,
                 'HMM_labels'             : float,
                 'labels'                 : object}

envelope_dtypes = {'L1 Erector Spinae LEFT' : float,
                   'L1 Erector Spinae RIGHT': float,
                   'L3 Erector Spinae LEFT' : float,
                   'L3 Erector Spinae RIGHT': float,
                   'L5 Multifidus LEFT'     : float,
                   'L5 Multifidus RIGHT'    : float,
                   'Rectus Abdominis RIGHT' : float,
                   'Rectus Abdominis LEFT'  : float,
                   'External Oblique RIGHT' : float,
                   'External Oblique LEFT'  : float,
                   'Gluteus Medius LEFT'    : float,
                   'Gluteus Medius RIGHT'   : float,
                   'Vastus Lateralis RIGHT' : float,
                   'Vastus Lateralis LEFT'  : float,
                   'Biceps Femoris LEFT'    : float,
                   'Biceps Femoris RIGHT'   : float,
                   'time'                   : float,
                   'HMM_labels'             : float,
                   'labels'                 : object}

#Rename function
def remove_number(string):
    if ':' in string:
        parts = string.split(':')
        return parts[0]
    else:
        return string

#Rename the column names w/ brackets
def remove_brackets(string):
    if '(' in string:
        A = re.sub("[\(\[].*?[\)\]]", "", string)
        B = re.sub("  ", " ", A)
        return B
    else:
        return string


xsens_dtypes = {'time_ms': float,
                'jL5S1_x': float,
                'jL5S1_y': float,
                'jL5S1_z': float,
                'jL4L3_x': float,
                'jL4L3_y': float,
                'jL4L3_z': float,
                'jL1T12_x': float,
                'jL1T12_y': float,
                'jL1T12_z': float,
                'jT9T8_x': float,
                'jT9T8_y': float,
                'jT9T8_z': float,
                'jT1C7_x': float,
                'jT1C7_y': float,
                'jT1C7_z': float,
                'jC1Head_x': float,
                'jC1Head_y': float,
                'jC1Head_z': float,
                'jRightT4Shoulder_x': float,
                'jRightT4Shoulder_y': float,
                'jRightT4Shoulder_z': float,
                'jRightShoulder_x': float,
                'jRightShoulder_y': float,
                'jRightShoulder_z': float,
                'jRightElbow_x': float,
                'jRightElbow_y': float,
                'jRightElbow_z': float,
                'jRightWrist_x': float,
                'jRightWrist_y': float,
                'jRightWrist_z': float,
                'jLeftT4Shoulder_x': float,
                'jLeftT4Shoulder_y': float,
                'jLeftT4Shoulder_z': float,
                'jLeftShoulder_x': float,
                'jLeftShoulder_y': float,
                'jLeftShoulder_z': float,
                'jLeftElbow_x': float,
                'jLeftElbow_y': float,
                'jLeftElbow_z': float,
                'jLeftWrist_x': float,
                'jLeftWrist_y': float,
                'jLeftWrist_z': float,
                'jRightHip_x': float,
                'jRightHip_y': float,
                'jRightHip_z': float,
                'jRightKnee_x': float,
                'jRightKnee_y': float,
                'jRightKnee_z': float,
                'jRightAnkle_x': float,
                'jRightAnkle_y': float,
                'jRightAnkle_z': float,
                'jRightBallFoot_x': float,
                'jRightBallFoot_y': float,
                'jRightBallFoot_z': float,
                'jLeftHip_x': float,
                'jLeftHip_y': float,
                'jLeftHip_z': float,
                'jLeftKnee_x': float,
                'jLeftKnee_y': float,
                'jLeftKnee_z': float,
                'jLeftAnkle_x': float,
                'jLeftAnkle_y': float,
                'jLeftAnkle_z': float,
                'jLeftBallFoot_x': float,
                'jLeftBallFoot_y': float,
                'jLeftBallFoot_z': float,
                'T8_Head_x': float,
                'T8_Head_y': float,
                'T8_Head_z': float,
                'T8_LeftUpperArm_x': float,
                'T8_LeftUpperArm_y': float,
                'T8_LeftUpperArm_z': float,
                'T8_RightUpperArm_x': float,
                'T8_RightUpperArm_y': float,
                'T8_RightUpperArm_z': float,
                'Pelvis_T8_x': float,
                'Pelvis_T8_y': float,
                'Pelvis_T8_z': float,
                'Vertical_Pelvis_x': float,
                'Vertical_Pelvis_y': float,
                'Vertical_Pelvis_z': float,
                'Vertical_T8_x': float,
                'Vertical_T8_y': float,
                'Vertical_T8_z': float,
                'CoG_x': float,
                'CoG_y': float,
                'CoG_z': float,
                'timecode': object,
                'HMM_labels': float,
                'labels': object,
                'LeftFoot_Heel': float,
                'LeftFoot_Toe': float,
                'RightFoot_Heel': float,
                'RightFoot_Toe': float,
                'Pelvis_acc_x': float,
                'Pelvis_acc_y': float,
                'Pelvis_acc_z': float,
                'T8_acc_x': float,
                'T8_acc_y': float,
                'T8_acc_z': float,
                'Head_acc_x': float,
                'Head_acc_y': float,
                'Head_acc_z': float,
                'RightShoulder_acc_x': float,
                'RightShoulder_acc_y': float,
                'RightShoulder_acc_z': float,
                'RightUpperArm_acc_x': float,
                'RightUpperArm_acc_y': float,
                'RightUpperArm_acc_z': float,
                'RightForeArm_acc_x': float,
                'RightForeArm_acc_y': float,
                'RightForeArm_acc_z': float,
                'LeftShoulder_acc_x': float,
                'LeftShoulder_acc_y': float,
                'LeftShoulder_acc_z': float,
                'LeftUpperArm_acc_x': float,
                'LeftUpperArm_acc_y': float,
                'LeftUpperArm_acc_z': float,
                'LeftForeArm_acc_x': float,
                'LeftForeArm_acc_y': float,
                'LeftForeArm_acc_z': float,
                'RightUpperLeg_acc_x': float,
                'RightUpperLeg_acc_y': float,
                'RightUpperLeg_acc_z': float,
                'RightLowerLeg_acc_x': float,
                'RightLowerLeg_acc_y': float,
                'RightLowerLeg_acc_z': float,
                'RightFoot_acc_x': float,
                'RightFoot_acc_y': float,
                'RightFoot_acc_z': float,
                'LeftUpperLeg_acc_x': float,
                'LeftUpperLeg_acc_y': float,
                'LeftUpperLeg_acc_z': float,
                'LeftLowerLeg_acc_x': float,
                'LeftLowerLeg_acc_y': float,
                'LeftLowerLeg_acc_z': float,
                'LeftFoot_acc_x': float,
                'LeftFoot_acc_y': float,
                'LeftFoot_acc_z': float,
                'Pelvis_orient_q1': float,
                'Pelvis_orient_q2': float,
                'Pelvis_orient_q3': float,
                'Pelvis_orient_q4': float,
                'Pelvis_pos_x': float,
                'Pelvis_pos_y': float,
                'Pelvis_pos_z': float,
                'Pelvis_vel_x': float,
                'Pelvis_vel_y': float,
                'Pelvis_vel_z': float,
                'Pelvis_acc_x.1': float,
                'Pelvis_acc_y.1': float,
                'Pelvis_acc_z.1': float,
                'Pelvis_angvel_x': float,
                'Pelvis_angvel_y': float,
                'Pelvis_angvel_z': float,
                'Pelvis_angacc_x': float,
                'Pelvis_angacc_y': float,
                'Pelvis_angacc_z': float,
                'L5_orient_q1': float,
                'L5_orient_q2': float,
                'L5_orient_q3': float,
                'L5_orient_q4': float,
                'L5_pos_x': float,
                'L5_pos_y': float,
                'L5_pos_z': float,
                'L5_vel_x': float,
                'L5_vel_y': float,
                'L5_vel_z': float,
                'L5_acc_x': float,
                'L5_acc_y': float,
                'L5_acc_z': float,
                'L5_angvel_x': float,
                'L5_angvel_y': float,
                'L5_angvel_z': float,
                'L5_angacc_x': float,
                'L5_angacc_y': float,
                'L5_angacc_z': float,
                'L3_orient_q1': float,
                'L3_orient_q2': float,
                'L3_orient_q3': float,
                'L3_orient_q4': float,
                'L3_pos_x': float,
                'L3_pos_y': float,
                'L3_pos_z': float,
                'L3_vel_x': float,
                'L3_vel_y': float,
                'L3_vel_z': float,
                'L3_acc_x': float,
                'L3_acc_y': float,
                'L3_acc_z': float,
                'L3_angvel_x': float,
                'L3_angvel_y': float,
                'L3_angvel_z': float,
                'L3_angacc_x': float,
                'L3_angacc_y': float,
                'L3_angacc_z': float,
                'T12_orient_q1': float,
                'T12_orient_q2': float,
                'T12_orient_q3': float,
                'T12_orient_q4': float,
                'T12_pos_x': float,
                'T12_pos_y': float,
                'T12_pos_z': float,
                'T12_vel_x': float,
                'T12_vel_y': float,
                'T12_vel_z': float,
                'T12_acc_x': float,
                'T12_acc_y': float,
                'T12_acc_z': float,
                'T12_angvel_x': float,
                'T12_angvel_y': float,
                'T12_angvel_z': float,
                'T12_angacc_x': float,
                'T12_angacc_y': float,
                'T12_angacc_z': float,
                'T8_orient_q1': float,
                'T8_orient_q2': float,
                'T8_orient_q3': float,
                'T8_orient_q4': float,
                'T8_pos_x': float,
                'T8_pos_y': float,
                'T8_pos_z': float,
                'T8_vel_x': float,
                'T8_vel_y': float,
                'T8_vel_z': float,
                'T8_acc_x.1': float,
                'T8_acc_y.1': float,
                'T8_acc_z.1': float,
                'T8_angvel_x': float,
                'T8_angvel_y': float,
                'T8_angvel_z': float,
                'T8_angacc_x': float,
                'T8_angacc_y': float,
                'T8_angacc_z': float,
                'Neck_orient_q1': float,
                'Neck_orient_q2': float,
                'Neck_orient_q3': float,
                'Neck_orient_q4': float,
                'Neck_pos_x': float,
                'Neck_pos_y': float,
                'Neck_pos_z': float,
                'Neck_vel_x': float,
                'Neck_vel_y': float,
                'Neck_vel_z': float,
                'Neck_acc_x': float,
                'Neck_acc_y': float,
                'Neck_acc_z': float,
                'Neck_angvel_x': float,
                'Neck_angvel_y': float,
                'Neck_angvel_z': float,
                'Neck_angacc_x': float,
                'Neck_angacc_y': float,
                'Neck_angacc_z': float,
                'Head_orient_q1': float,
                'Head_orient_q2': float,
                'Head_orient_q3': float,
                'Head_orient_q4': float,
                'Head_pos_x': float,
                'Head_pos_y': float,
                'Head_pos_z': float,
                'Head_vel_x': float,
                'Head_vel_y': float,
                'Head_vel_z': float,
                'Head_acc_x.1': float,
                'Head_acc_y.1': float,
                'Head_acc_z.1': float,
                'Head_angvel_x': float,
                'Head_angvel_y': float,
                'Head_angvel_z': float,
                'Head_angacc_x': float,
                'Head_angacc_y': float,
                'Head_angacc_z': float,
                'RightShoulder_orient_q1': float,
                'RightShoulder_orient_q2': float,
                'RightShoulder_orient_q3': float,
                'RightShoulder_orient_q4': float,
                'RightShoulder_pos_x': float,
                'RightShoulder_pos_y': float,
                'RightShoulder_pos_z': float,
                'RightShoulder_vel_x': float,
                'RightShoulder_vel_y': float,
                'RightShoulder_vel_z': float,
                'RightShoulder_acc_x.1': float,
                'RightShoulder_acc_y.1': float,
                'RightShoulder_acc_z.1': float,
                'RightShoulder_angvel_x': float,
                'RightShoulder_angvel_y': float,
                'RightShoulder_angvel_z': float,
                'RightShoulder_angacc_x': float,
                'RightShoulder_angacc_y': float,
                'RightShoulder_angacc_z': float,
                'RightUpperArm_orient_q1': float,
                'RightUpperArm_orient_q2': float,
                'RightUpperArm_orient_q3': float,
                'RightUpperArm_orient_q4': float,
                'RightUpperArm_pos_x': float,
                'RightUpperArm_pos_y': float,
                'RightUpperArm_pos_z': float,
                'RightUpperArm_vel_x': float,
                'RightUpperArm_vel_y': float,
                'RightUpperArm_vel_z': float,
                'RightUpperArm_acc_x.1': float,
                'RightUpperArm_acc_y.1': float,
                'RightUpperArm_acc_z.1': float,
                'RightUpperArm_angvel_x': float,
                'RightUpperArm_angvel_y': float,
                'RightUpperArm_angvel_z': float,
                'RightUpperArm_angacc_x': float,
                'RightUpperArm_angacc_y': float,
                'RightUpperArm_angacc_z': float,
                'RightForeArm_orient_q1': float,
                'RightForeArm_orient_q2': float,
                'RightForeArm_orient_q3': float,
                'RightForeArm_orient_q4': float,
                'RightForeArm_pos_x': float,
                'RightForeArm_pos_y': float,
                'RightForeArm_pos_z': float,
                'RightForeArm_vel_x': float,
                'RightForeArm_vel_y': float,
                'RightForeArm_vel_z': float,
                'RightForeArm_acc_x.1': float,
                'RightForeArm_acc_y.1': float,
                'RightForeArm_acc_z.1': float,
                'RightForeArm_angvel_x': float,
                'RightForeArm_angvel_y': float,
                'RightForeArm_angvel_z': float,
                'RightForeArm_angacc_x': float,
                'RightForeArm_angacc_y': float,
                'RightForeArm_angacc_z': float,
                'RightHand_orient_q1': float,
                'RightHand_orient_q2': float,
                'RightHand_orient_q3': float,
                'RightHand_orient_q4': float,
                'RightHand_pos_x': float,
                'RightHand_pos_y': float,
                'RightHand_pos_z': float,
                'RightHand_vel_x': float,
                'RightHand_vel_y': float,
                'RightHand_vel_z': float,
                'RightHand_acc_x': float,
                'RightHand_acc_y': float,
                'RightHand_acc_z': float,
                'RightHand_angvel_x': float,
                'RightHand_angvel_y': float,
                'RightHand_angvel_z': float,
                'RightHand_angacc_x': float,
                'RightHand_angacc_y': float,
                'RightHand_angacc_z': float,
                'LeftShoulder_orient_q1': float,
                'LeftShoulder_orient_q2': float,
                'LeftShoulder_orient_q3': float,
                'LeftShoulder_orient_q4': float,
                'LeftShoulder_pos_x': float,
                'LeftShoulder_pos_y': float,
                'LeftShoulder_pos_z': float,
                'LeftShoulder_vel_x': float,
                'LeftShoulder_vel_y': float,
                'LeftShoulder_vel_z': float,
                'LeftShoulder_acc_x.1': float,
                'LeftShoulder_acc_y.1': float,
                'LeftShoulder_acc_z.1': float,
                'LeftShoulder_angvel_x': float,
                'LeftShoulder_angvel_y': float,
                'LeftShoulder_angvel_z': float,
                'LeftShoulder_angacc_x': float,
                'LeftShoulder_angacc_y': float,
                'LeftShoulder_angacc_z': float,
                'LeftUpperArm_orient_q1': float,
                'LeftUpperArm_orient_q2': float,
                'LeftUpperArm_orient_q3': float,
                'LeftUpperArm_orient_q4': float,
                'LeftUpperArm_pos_x': float,
                'LeftUpperArm_pos_y': float,
                'LeftUpperArm_pos_z': float,
                'LeftUpperArm_vel_x': float,
                'LeftUpperArm_vel_y': float,
                'LeftUpperArm_vel_z': float,
                'LeftUpperArm_acc_x.1': float,
                'LeftUpperArm_acc_y.1': float,
                'LeftUpperArm_acc_z.1': float,
                'LeftUpperArm_angvel_x': float,
                'LeftUpperArm_angvel_y': float,
                'LeftUpperArm_angvel_z': float,
                'LeftUpperArm_angacc_x': float,
                'LeftUpperArm_angacc_y': float,
                'LeftUpperArm_angacc_z': float,
                'LeftForeArm_orient_q1': float,
                'LeftForeArm_orient_q2': float,
                'LeftForeArm_orient_q3': float,
                'LeftForeArm_orient_q4': float,
                'LeftForeArm_pos_x': float,
                'LeftForeArm_pos_y': float,
                'LeftForeArm_pos_z': float,
                'LeftForeArm_vel_x': float,
                'LeftForeArm_vel_y': float,
                'LeftForeArm_vel_z': float,
                'LeftForeArm_acc_x.1': float,
                'LeftForeArm_acc_y.1': float,
                'LeftForeArm_acc_z.1': float,
                'LeftForeArm_angvel_x': float,
                'LeftForeArm_angvel_y': float,
                'LeftForeArm_angvel_z': float,
                'LeftForeArm_angacc_x': float,
                'LeftForeArm_angacc_y': float,
                'LeftForeArm_angacc_z': float,
                'LeftHand_orient_q1': float,
                'LeftHand_orient_q2': float,
                'LeftHand_orient_q3': float,
                'LeftHand_orient_q4': float,
                'LeftHand_pos_x': float,
                'LeftHand_pos_y': float,
                'LeftHand_pos_z': float,
                'LeftHand_vel_x': float,
                'LeftHand_vel_y': float,
                'LeftHand_vel_z': float,
                'LeftHand_acc_x': float,
                'LeftHand_acc_y': float,
                'LeftHand_acc_z': float,
                'LeftHand_angvel_x': float,
                'LeftHand_angvel_y': float,
                'LeftHand_angvel_z': float,
                'LeftHand_angacc_x': float,
                'LeftHand_angacc_y': float,
                'LeftHand_angacc_z': float,
                'RightUpperLeg_orient_q1': float,
                'RightUpperLeg_orient_q2': float,
                'RightUpperLeg_orient_q3': float,
                'RightUpperLeg_orient_q4': float,
                'RightUpperLeg_pos_x': float,
                'RightUpperLeg_pos_y': float,
                'RightUpperLeg_pos_z': float,
                'RightUpperLeg_vel_x': float,
                'RightUpperLeg_vel_y': float,
                'RightUpperLeg_vel_z': float,
                'RightUpperLeg_acc_x.1': float,
                'RightUpperLeg_acc_y.1': float,
                'RightUpperLeg_acc_z.1': float,
                'RightUpperLeg_angvel_x': float,
                'RightUpperLeg_angvel_y': float,
                'RightUpperLeg_angvel_z': float,
                'RightUpperLeg_angacc_x': float,
                'RightUpperLeg_angacc_y': float,
                'RightUpperLeg_angacc_z': float,
                'RightLowerLeg_orient_q1': float,
                'RightLowerLeg_orient_q2': float,
                'RightLowerLeg_orient_q3': float,
                'RightLowerLeg_orient_q4': float,
                'RightLowerLeg_pos_x': float,
                'RightLowerLeg_pos_y': float,
                'RightLowerLeg_pos_z': float,
                'RightLowerLeg_vel_x': float,
                'RightLowerLeg_vel_y': float,
                'RightLowerLeg_vel_z': float,
                'RightLowerLeg_acc_x.1': float,
                'RightLowerLeg_acc_y.1': float,
                'RightLowerLeg_acc_z.1': float,
                'RightLowerLeg_angvel_x': float,
                'RightLowerLeg_angvel_y': float,
                'RightLowerLeg_angvel_z': float,
                'RightLowerLeg_angacc_x': float,
                'RightLowerLeg_angacc_y': float,
                'RightLowerLeg_angacc_z': float,
                'RightFoot_orient_q1': float,
                'RightFoot_orient_q2': float,
                'RightFoot_orient_q3': float,
                'RightFoot_orient_q4': float,
                'RightFoot_pos_x': float,
                'RightFoot_pos_y': float,
                'RightFoot_pos_z': float,
                'RightFoot_vel_x': float,
                'RightFoot_vel_y': float,
                'RightFoot_vel_z': float,
                'RightFoot_acc_x.1': float,
                'RightFoot_acc_y.1': float,
                'RightFoot_acc_z.1': float,
                'RightFoot_angvel_x': float,
                'RightFoot_angvel_y': float,
                'RightFoot_angvel_z': float,
                'RightFoot_angacc_x': float,
                'RightFoot_angacc_y': float,
                'RightFoot_angacc_z': float,
                'RightToe_orient_q1': float,
                'RightToe_orient_q2': float,
                'RightToe_orient_q3': float,
                'RightToe_orient_q4': float,
                'RightToe_pos_x': float,
                'RightToe_pos_y': float,
                'RightToe_pos_z': float,
                'RightToe_vel_x': float,
                'RightToe_vel_y': float,
                'RightToe_vel_z': float,
                'RightToe_acc_x': float,
                'RightToe_acc_y': float,
                'RightToe_acc_z': float,
                'RightToe_angvel_x': float,
                'RightToe_angvel_y': float,
                'RightToe_angvel_z': float,
                'RightToe_angacc_x': float,
                'RightToe_angacc_y': float,
                'RightToe_angacc_z': float,
                'LeftUpperLeg_orient_q1': float,
                'LeftUpperLeg_orient_q2': float,
                'LeftUpperLeg_orient_q3': float,
                'LeftUpperLeg_orient_q4': float,
                'LeftUpperLeg_pos_x': float,
                'LeftUpperLeg_pos_y': float,
                'LeftUpperLeg_pos_z': float,
                'LeftUpperLeg_vel_x': float,
                'LeftUpperLeg_vel_y': float,
                'LeftUpperLeg_vel_z': float,
                'LeftUpperLeg_acc_x.1': float,
                'LeftUpperLeg_acc_y.1': float,
                'LeftUpperLeg_acc_z.1': float,
                'LeftUpperLeg_angvel_x': float,
                'LeftUpperLeg_angvel_y': float,
                'LeftUpperLeg_angvel_z': float,
                'LeftUpperLeg_angacc_x': float,
                'LeftUpperLeg_angacc_y': float,
                'LeftUpperLeg_angacc_z': float,
                'LeftLowerLeg_orient_q1': float,
                'LeftLowerLeg_orient_q2': float,
                'LeftLowerLeg_orient_q3': float,
                'LeftLowerLeg_orient_q4': float,
                'LeftLowerLeg_pos_x': float,
                'LeftLowerLeg_pos_y': float,
                'LeftLowerLeg_pos_z': float,
                'LeftLowerLeg_vel_x': float,
                'LeftLowerLeg_vel_y': float,
                'LeftLowerLeg_vel_z': float,
                'LeftLowerLeg_acc_x.1': float,
                'LeftLowerLeg_acc_y.1': float,
                'LeftLowerLeg_acc_z.1': float,
                'LeftLowerLeg_angvel_x': float,
                'LeftLowerLeg_angvel_y': float,
                'LeftLowerLeg_angvel_z': float,
                'LeftLowerLeg_angacc_x': float,
                'LeftLowerLeg_angacc_y': float,
                'LeftLowerLeg_angacc_z': float,
                'LeftFoot_orient_q1': float,
                'LeftFoot_orient_q2': float,
                'LeftFoot_orient_q3': float,
                'LeftFoot_orient_q4': float,
                'LeftFoot_pos_x': float,
                'LeftFoot_pos_y': float,
                'LeftFoot_pos_z': float,
                'LeftFoot_vel_x': float,
                'LeftFoot_vel_y': float,
                'LeftFoot_vel_z': float,
                'LeftFoot_acc_x.1': float,
                'LeftFoot_acc_y.1': float,
                'LeftFoot_acc_z.1': float,
                'LeftFoot_angvel_x': float,
                'LeftFoot_angvel_y': float,
                'LeftFoot_angvel_z': float,
                'LeftFoot_angacc_x': float,
                'LeftFoot_angacc_y': float,
                'LeftFoot_angacc_z': float,
                'LeftToe_orient_q1': float,
                'LeftToe_orient_q2': float,
                'LeftToe_orient_q3': float,
                'LeftToe_orient_q4': float,
                'LeftToe_pos_x': float,
                'LeftToe_pos_y': float,
                'LeftToe_pos_z': float,
                'LeftToe_vel_x': float,
                'LeftToe_vel_y': float,
                'LeftToe_vel_z': float,
                'LeftToe_acc_x': float,
                'LeftToe_acc_y': float,
                'LeftToe_acc_z': float,
                'LeftToe_angvel_x': float,
                'LeftToe_angvel_y': float,
                'LeftToe_angvel_z': float,
                'LeftToe_angacc_x': float,
                'LeftToe_angacc_y': float,
                'LeftToe_angacc_z': float}