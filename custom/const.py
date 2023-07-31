task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
    'etype':['EVENT','TIMEX3','SECTIME'],
    'ner': ['CLINICAL_DEPT', 'EVIDENTIAL', 'OCCURRENCE', 'PROBLEM', 'TEST','TREATMENT', 'TIMEX3', 'DISCHARGE', 'ADMISSION'],
    'ner_plus_time': ['CLINICAL_DEPT', 'EVIDENTIAL', 'OCCURRENCE', 'PROBLEM', 'TEST','TREATMENT', 'TIMEX3', 'DISCHARGE', 'ADMISSION', 'DATE', 'DURATION', 'FREQUENCY', 'TIME']
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PER-SOC', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
    'i2b2': ['BEFORE', 'AFTER', 'OVERLAP'],
    'tbd': ['SIMULTANEOUS', 'IS_INCLUDED', 'VAGUE', 'INCLUDES', 'AFTER', 'BEFORE']
}
 #'tbd': ['overlap', 'before', 'after', 'vague', 'includs', 'is_included'] # NOTE: directly from CTRL-PG, but now from TEPreprocess

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i + 1
        id2label[i + 1] = label
    return label2id, id2label
