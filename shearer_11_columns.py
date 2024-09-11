# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:08:49 2024

@author: robin
"""

import read_delsys as rd

A = ['1 - Gluteus Medius (Left): EMG 1', '2 - L1 ES Left: EMG 2',
       '3 - Vastus Lateralis (Right): EMG 3',
       '4 - Vastus Lateralis (Left): EMG 4', '5 - L1 ES Right: EMG 5',
       '6 - L3 ES Left: EMG 6', '7 - L3 ES Right: EMG 7',
       '8 - L5 MF Left: EMG 8', '9 - L5 MF Right: EMG 9',
       '10 - Rectus Abdominis Right: EMG 10',
       '11 - Rectus Abdominis Left: EMG 11',
       '12 - External Oblique Right: EMG 12',
       '13 - External Oblique Left: EMG 13',
       '14 - Gluteus Medius Right: EMG 14',
       '15 - Biceps femoris (Hamstring) left: EMG 15',
       '16 - Biceps femoris (hamstring) Right: EMG 16', 'time']


A_new =['Gluteus Medius LEFT: EMG.A 11', 'L1 Erector Spinae LEFT: EMG.A 1',
       'Vastus Lateralis RIGHT: EMG.A 13',
       'Vastus Lateralis LEFT: EMG.A 14', 'L1 Erector Spinae RIGHT: EMG.A 2',
       'L3 Erector Spinae LEFT: EMG.A 3', 'L3 Erector Spinae RIGHT: EMG.A 4',
       'L5 Multifidus LEFT: EMG.A 5', 'L5 Multifidus RIGHT: EMG.A 6',
       'Rectus Abdominis (1cm up, 3cm out) RIGHT: EMG.A 7',
       'Rectus Abdominis (1cm up, 3cm out) LEFT: EMG.A 8',
       'External Oblique (15cm out) RIGHT: EMG.A 9',
       'External Oblique (15cm out) LEFT: EMG.A 10',
       'Gluteus Medius RIGHT: EMG.A 12',
      'Biceps Femoris LEFT: EMG.A 15',
       'Biceps Femoris RIGHT: EMG.A 16', 'time']

shearer_11_redo = rd.read_delsys_csv("D:\Data_for_up\Week 4\s11_week4_friday\s11_week4_friday_run1\\s11_week4_friday_run1.csv")

for i in range(0,len(shearer_11_redo.columns)):
    assert(shearer_11_redo.columns[i] == A[i])

shearer_11_redo.columns = A_new

#with open("D:\Data_for_up\Week 4\s11_week4_friday\s11_week4_friday_run1\\s11_week4_friday_run1_emg.csv", 'w') as f:
#    shearer_11_redo.to_csv(f,index_label='sample')