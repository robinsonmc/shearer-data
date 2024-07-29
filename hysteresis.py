# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:25:28 2020

@author: mrobinson2
"""

def asym_hysteresis(result, long_th, short_th):
    import copy
    
    orig_result = copy.copy(result)
    i = 2
    latch_flag = 0
    while i < len(orig_result)-1:
        #Asymmetric de-bouncing with hysteresis
        if ((orig_result[i] != orig_result[i-1]) & (orig_result[i-1] == 0)):
            for j in range(long_th):
                if (i+j+1 < len(orig_result)) and (orig_result[i+j+1] != orig_result[i+j]):
                    latch_flag = 1
                    counter = j+1
                    break
            if latch_flag == 1:
                assert(len(orig_result[i:i+counter]) == len([orig_result[i-1]]*counter))
                orig_result[i:i+counter] = [orig_result[i-1]]*counter
                latch_flag = 0
                i+=counter
        elif ((orig_result[i] != orig_result[i-1]) & (orig_result[i-1] == 1)):
            for k in range(short_th):
                if (i+k+1 < len(orig_result)) and (orig_result[i+k+1] != orig_result[i+k]):
                    latch_flag = 1
                    counter = k+1
                    break
            if latch_flag == 1:
                assert(len(orig_result[i:i+counter]) == len([orig_result[i-1]]*counter))
                orig_result[i:i+counter] = [orig_result[i-1]]*counter
                latch_flag = 0
                i+=counter
        i += 1
    return orig_result

def label_data(feature_df, model):
    import numpy as np
    
    sequences = feature_df
    AA = np.array(sequences,dtype=float)
    BB = [list(x) for x in AA]
    
    result = model.predict(BB,algorithm='viterbi')
    orig_result = asym_hysteresis(result, 350, 80)
    return orig_result

def test_labels(labels,result):
    import matplotlib.pyplot as plt
    labels = list(labels)
    for i in range(len(labels)):
        if labels[i] == 'shearing':
            labels[i] = 1
        if labels[i] == 'catch_drag':
            labels[i] = 0
            
    plt.figure()
    plt.plot(labels)
    plt.plot(result)
    
def evaluate_result(labels,result):
    labels = list(labels)
    for i in range(len(labels)):
        if labels[i] == 'shearing':
            labels[i] = 1
        if labels[i] == 'catch_drag':
            labels[i] = 0
            
    if len(result) - len(labels) == 1:
        result = result[1:]
    elif len(result) - len(labels) > 1:
        result = result[1:-1]
        
    assert(len(result) == len(labels))
    
    true_shearing  = 0
    false_shearing = 0
    true_CD        = 0
    false_CD       = 0
    for i in range(len(result)):
        if result[i] == 1:
            if labels[i] == 1:
                true_shearing += 1
            else:
                false_shearing += 1
        elif result[i] == 0:
            if labels[i] == 0:
                true_CD += 1
            else:
                false_CD += 1
        else:
            raise Exception('Non-zero or one value found')
            
    print('Results:')
    print('Shearing precision = {:.2f}'.format(100*true_shearing/(true_shearing+false_shearing)))
    print('Shearing recall = {:.2f}'.format(100*true_shearing/(true_shearing+false_CD)))
    print('Catch and drag precision = {:.2f}'.format(100*true_CD/(true_CD+false_CD)))
    print('Catch and drag recall = {:.2f}'.format(100*true_CD/(true_CD+false_shearing)))
    print('Accuracy = {:.2f}'.format(100*(true_CD+true_shearing)/(true_CD+false_shearing+true_shearing+false_CD)))
    print('Shearing F-1 score = {:.2f}'.format(2*(100*true_shearing/(true_shearing+false_shearing)*100*true_shearing/(true_shearing+false_CD))/(100*true_shearing/(true_shearing+false_shearing)+100*true_shearing/(true_shearing+false_CD))))
    print('Catch and drag F-1 score = {:.2f}'.format(2*(100*true_CD/(true_CD+false_CD)*100*true_CD/(true_CD+false_shearing))/(100*true_CD/(true_CD+false_CD)+100*true_CD/(true_CD+false_shearing))))
    return (true_shearing, false_shearing, true_CD, false_CD)

def evaluate_result_LOPO(labels,result):
    labels = list(labels)
    for i in range(len(labels)):
        if labels[i] == 'shearing':
            labels[i] = 1
        if labels[i] == 'catch_drag':
            labels[i] = 0
            
    if len(result) - len(labels) == 1:
        result = result[1:]
    elif len(result) - len(labels) > 1:
        result = result[1:-1]
        
    assert(len(result) == len(labels))
    
    true_shearing  = 0
    false_shearing = 0
    true_CD        = 0
    false_CD       = 0
    for i in range(len(result)):
        if result[i] == 1:
            if labels[i] == 1:
                true_shearing += 1
            else:
                false_shearing += 1
        elif result[i] == 0:
            if labels[i] == 0:
                true_CD += 1
            else:
                false_CD += 1
        else:
            raise Exception('Non-zero or one value found')
            
    return (true_shearing, false_shearing, true_CD, false_CD)

def shearing_precision(true_shearing, false_shearing, true_CD, false_CD):
    return 100*true_shearing/(true_shearing+false_shearing)

def shearing_recall(true_shearing, false_shearing, true_CD, false_CD):
    return 100*true_shearing/(true_shearing+false_CD)

def catch_drag_precision(true_shearing, false_shearing, true_CD, false_CD):
    return 100*true_CD/(true_CD+false_CD)

def catch_drag_recall(true_shearing, false_shearing, true_CD, false_CD):
    return 100*true_CD/(true_CD+false_shearing)

def accuracy(true_shearing, false_shearing, true_CD, false_CD):
    return 100*(true_CD+true_shearing)/(true_CD+false_shearing+true_shearing+false_CD)

def post_process(result):
    #Post-process
    
    #Get rid of whatever is between two long catch drag >1800 samples 90 seconds
    #-----------
    
    #Cut off short catch drag at <150 samples 7.5 seconds - try 200
    #Shortest legit drag was 242 samples - 12 seconds
    for i in range(1,len(result)):
        if result[i] == 0 and result[i-1] == 1 and i < len(result)-200:
            for j in range(200):
                if result[i+j] == 1:
                    result[i:i+j] = [1-x for x in result[i:i+j]]
                    i = i+j
                    break
                
    #Get rid of really short shearing <600 samples 20 seconds (750 too long)
    #Shortest legit shear was 1311 samples - 66 seconds
    for i in range(1,len(result)):
        if result[i] == 1 and result[i-1] == 0 and i < len(result)-550:
            for j in range(550):
                if result[i+j] == 0:
                    result[i:i+j] = [1-x for x in result[i:i+j]]
                    i = i+j
                    break
                
    #Get rid of whatever is between two shearing sessions when one is shorter
    #than the world record pace ~50s <1000 samples
    #try:
    for i in range(1,len(result)):
        try:
            if result[i] == 1 and result[i-1] == 0: #Put in try after testing
                #print('start shearing {}...'.format(i))
                CSF = 0
                j = 0
                while result[i+j] == result[i+j+1]:
                    CSF += 1
                    j += 1
                #print(CSF)
                CCM = 0
                j+=1
                while result[i+j] == result[i+j+1]:
                    CCM += 1
                    j += 1
                #print(CCM)
                CSS = 0
                j+=1
                while result[i+j] == result[i+j+1]:
                    CSS += 1
                    j += 1
                #print(CSS)
                if CSF < 1200 and CSS < 1200 and (CSF+CSS > 1200):
                    result[i+CSF+1:i+CSF+CCM+2] = [1-x for x in result[i+CSF+1:i+CSF+CCM+2]]
                i=i+j
            
        except IndexError:
            break
            
    return result

def new_score(labels,result):
    '''
    Given a set of labels and results implement the frame labelling from
    Ward et al (2011) metric for scoring activity recognition
    '''
    from copy import copy
    labels = list(labels)
    for i in range(len(labels)):
        if labels[i] == 'shearing':
            labels[i] = 1
        if labels[i] == 'catch_drag':
            labels[i] = 0
            
    if len(result) - len(labels) == 1:
        result = result[1:]
    elif len(result) - len(labels) > 1:
        result = result[1:-1]
        
    assert(len(result) == len(labels))
    
    r_label = []
    for i in range(len(result)):
        if result[i] == 1:
            if labels[i] == 1:
                r_label.append('TP')
            else:
                r_label.append('FP')
        elif result[i] == 0:
            if labels[i] == 0:
                r_label.append('TN')
            else:
                r_label.append('FN')
        else:
            raise Exception('Non-zero or one value found')
    
    r_label.append('END')
    #Assign each frame based on classification
    last_segment = 'START'
    final = copy(r_label)
    i = 0
    while i < len(r_label):
        j = 0
        if r_label[i+j] == 'END': break
        while r_label[i+j] == r_label[i+j+1]:
            j+=1
        temp_last_segment = r_label[i+j]
        seg = classify(last_segment,r_label[i+j],r_label[i+j+1])
        final[i:i+j+1] = [seg for x in final[i:i+j+1]]
        last_segment = temp_last_segment
        i = i+j+1
    return count_of(final[:-1])
        
        
def classify(F,M,L):
    encoding = {'START': '0', 'FP': '1', 'FN': '2', 'TP': '3', 'TN': '4',\
                'END': '5'}
    lookup = {'014': 'M','012': 'M','024': 'D','021': 'D','013': 'Os',\
              '023': 'Us','313': 'M','323': 'F','413': 'Os','213': 'Os',\
              '423': 'Us','123': 'Us','414': 'I','214': 'I','412': 'I',\
              '212': 'I','424': 'D','124': 'D','121': 'D','421': 'D',\
              '314': 'Oe','312': 'Oe','324': 'Ue','321': 'Ue','415': 'I',\
              '215': 'I','425': 'D','125': 'D','315': 'Oe','325': 'Ue'}
    if encoding[M] == '3': 
        return 'TP'
    if encoding[M] == '4':
        return 'TN'
    return lookup[encoding[F] + encoding[M] + encoding[L]]

def count_of(final):
    TP_count = final.count('TP')
    TN_count = final.count('TN')
    M_count  = final.count('M')
    I_count  = final.count('I')
    D_count  = final.count('D')
    F_count  = final.count('F')
    U_count  = final.count('Us') + final.count('Ue')
    O_count  = final.count('Os') + final.count('Oe')
    
    result_dict = {}
    
    result_dict['TP'] = TP_count
    result_dict['TN'] = TN_count
    result_dict['M'] = M_count
    result_dict['I'] = I_count
    result_dict['D'] = D_count
    result_dict['F'] = F_count
    result_dict['U'] = U_count
    result_dict['O'] = O_count
    result_dict['total'] = len(final)
    return result_dict
        
    
    