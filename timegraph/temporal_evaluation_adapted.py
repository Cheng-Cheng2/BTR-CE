#!/usr/bin/python 

'''
This TLINK evaluation script is written by Naushad UzZaman for TempEvel 3 evaluation

It is adapted by Weiyi Sun to fit the i2b2 xml format and annotation guidelines.

The changes include:
    get_relation(): modified to fit the i2b2 format
    get_relation_from_dictionary(): added to unify extent ids in the system/gold xmls
    evaluate_two_files(): modified accordingly


'''
# this program evaluates systems that extract temporal information from text 
# tlink -> temporal links

#foreach f (24-a-gold-tlinks/data/ABC19980108.1830.0711.tml); do
#python evaluation-relations/code/temporal_evaluation.py $f $(echo $f | p 's/24-a-gold-tlinks/30-b-trips-relations/g')                             
#done

# DURING relations are changed to SIMULTANEOUS



import time 
import sys
import re 
import os
'''
def get_arg (index):
    #for arg in sys.argv:
    return sys.argv[index]
'''
global_prec_matched = 0 
global_rec_matched = 0 
global_system_total = 0 
global_gold_total = 0 
'''
basedir = re.sub('relation_to_timegraph.py', '', get_arg(0)) 
#debug = int(get_arg(1))
debug=1
cmd_folder = os.path.dirname(basedir)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
    
'''
debug=0
sys.path.append('/data/chengc7/MarkerTRel/timegraph')
from relation_to_timegraph import interval_rel_X_Y, Timegraph, create_timegraph_from_weight_sorted_relations

    # tg_gold = relation_to_timegraph.Timegraph() 
    # tg_gold = relation_to_timegraph.create_timegraph_from_weight_sorted_relations(gold_text, tg_gold) 
    # tg_gold.final_relations = tg_gold.final_relations + tg_gold.violated_relations
    # tg_system = relation_to_timegraph.Timegraph() 
    # tg_system = relation_to_timegraph.create_timegraph_from_weight_sorted_relations(system_text, tg_system) 
# split based on link type
def event_is_e(r_str, eid):
    eid=eid.lower()
    search_result = re.search(r_str, eid)
    if search_result is None:
        return False
    if re.search(r_str, eid).group(0)==eid:
        return True
    return False
def extract_name(filename):
    parts = re.split('/', filename)
    length = len(parts)
    return parts[length-1]

def get_directory_path(path): 
    name = extract_name(path)
    dir = re.sub(name, '', path) 
    if dir == '': 
        dir = './'
    return dir 


def get_entity_val(word, line): 
    if re.search(word+'="[^"]*"', line): 
        entity = re.findall(word+'="[^"]*"', line)[0]
        entity = re.sub(word+'=', '', entity) 
        entity = re.sub('"', '', entity) 
        return entity 
    return word 
        
def change_DURING_relation(filetext): 
    newtext = '' 
    for line in filetext.split('\n'): 
        foo = '' 
        words = line.split('\t') 
        for i in range(0, len(words)): 
            if i == 3 and words[i] == 'DURING': 
                foo += re.sub('DURING', 'SIMULTANEOUS', words[i]) + '\t'
            else:
                foo += words[i] + '\t' 
        newtext += foo.strip() + '\n' 
    return newtext 

def get_relations(tlink_xml,dicsys,dic): 
    '''
    text = open(file).read()
    
    newtext = '' 
    name = extract_name(file) 
    relations = re.findall('<TLINK[^>]*>', text) 
    for each in relations: 
        core = '' 
        ref = '' 
        relType = '' 
        if re.search('eventInstanceID', each): 
            core = get_entity_val('eventInstanceID', each) 
        if re.search('timeID', each): 
            core = get_entity_val('timeID', each) 
        if re.search('relatedToEventInstance', each): 
            ref = get_entity_val('relatedToEventInstance', each) 
        if re.search('relatedToTime', each): 
            ref = get_entity_val('relatedToTime', each) 
        if re.search('relType', each): 
            relType = get_entity_val('relType', each) 
        if core == '' or ref == '' or relType == '': 
            print 'MISSING core, ref or relation', each 
    '''
    # read our xml file instead
    tlinklines=open(tlink_xml).readlines()
    newtext=''
    existing_ids={}
    for tlinkline in tlinklines:
        if re.search('<TLINK', tlinkline):
            re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+.*\/>'
            m = re.search(re_exp, tlinkline)
            if m:
                id, core, fromtext, ref, totext, relType = m.groups()
            else:
                raise Exception("Malformed EVENT tag: %s" % (tlinkline))
            dicsys['Admission']='Admission'
            dicsys['Discharge']='Discharge'
            
            if core!='' and ref!='' and relType!='':
                if len(dicsys)<3:
                    test_core=core
                    test_ref=ref
                else:
                    
                    try:
                        test_core_tuple=dicsys[core]
                        test_core=test_core_tuple.split('#@#')[0]
                    except KeyError:
                        test_core=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % core)
                    try:
                        test_ref_tuple=dicsys[ref]
                        test_ref=test_ref_tuple.split('#@#')[0]      
                    except KeyError:
                        test_ref=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % ref)
                if core!='' and ref!='':
                        
                    try:
                        core=existing_ids[test_core]
                    except KeyError:
                        try:
                            core=dic[test_core].split('#@#')[0]
                            existing_ids[test_core]=core   
                        except KeyError:
                            pass
                    try:
                        ref=existing_ids[test_ref]
                    except KeyError:
                        try:
                            ref=dic[test_ref].split('#@#')[0]
                            existing_ids[test_ref]=ref
                        except KeyError:
                            pass 
                    relType=relType.replace('OVERLAP','SIMULTANEOUS')
                    foo = tlink_xml+'\t'+core+'\t'+ref+'\t'+relType+'\n'
                    if debug >= 3: 
                        print(foo) 
                    newtext += foo 

    #newtext = change_DURING_relation(newtext)
    return newtext


def get_relations_plus_ids(tlink_xml,dicsys,dic): 
    '''
    text = open(file).read()
    
    newtext = '' 
    name = extract_name(file) 
    relations = re.findall('<TLINK[^>]*>', text) 
    for each in relations: 
        core = '' 
        ref = '' 
        relType = '' 
        if re.search('eventInstanceID', each): 
            core = get_entity_val('eventInstanceID', each) 
        if re.search('timeID', each): 
            core = get_entity_val('timeID', each) 
        if re.search('relatedToEventInstance', each): 
            ref = get_entity_val('relatedToEventInstance', each) 
        if re.search('relatedToTime', each): 
            ref = get_entity_val('relatedToTime', each) 
        if re.search('relType', each): 
            relType = get_entity_val('relType', each) 
        if core == '' or ref == '' or relType == '': 
            print 'MISSING core, ref or relation', each 
    '''
    # read our xml file instead
    tlinklines=open(tlink_xml).readlines()
    newtext=''
    existing_ids={}
    track_ids = {}
    for tlinkline in tlinklines:
        if re.search('<TLINK', tlinkline):
            re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+.*\/>'
            m = re.search(re_exp, tlinkline)
            if m:
                id, core, fromtext, ref, totext, relType = m.groups()
            else:
                raise Exception("Malformed EVENT tag: %s" % (tlinkline))
            dicsys['Admission']='Admission'
            dicsys['Discharge']='Discharge'
            
            if core!='' and ref!='' and relType!='':
                if len(dicsys)<3:
                    test_core=core
                    test_ref=ref
                else:
                    
                    try:
                        test_core_tuple=dicsys[core]
                        test_core=test_core_tuple.split('#@#')[0]
                    except KeyError:
                        test_core=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % core)
                    try:
                        test_ref_tuple=dicsys[ref]
                        test_ref=test_ref_tuple.split('#@#')[0]      
                    except KeyError:
                        test_ref=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % ref)
                if core!='' and ref!='':
                        
                    try:
                        core=existing_ids[test_core]
                    except KeyError:
                        try:
                            core=dic[test_core].split('#@#')[0]
                            existing_ids[test_core]=core   
                        except KeyError:
                            pass
                    try:
                        ref=existing_ids[test_ref]
                    except KeyError:
                        try:
                            ref=dic[test_ref].split('#@#')[0]
                            existing_ids[test_ref]=ref
                        except KeyError:
                            pass 
                    relType=relType.replace('OVERLAP','SIMULTANEOUS')
                    foo = tlink_xml+'\t'+core+'\t'+ref+'\t'+relType+'\n'
                    if debug >= 3: 
                        print(foo) 
                    newtext += foo 
                    track_ids[(core, ref)]=id
                    

    #newtext = change_DURING_relation(newtext)
    return newtext, track_ids

def get_relations_from_dictionary_plus_ids(tlink_xml,dic):
    tlinklines=open(tlink_xml).readlines()
    newtext=''
    existing_ids={}
    count=0
    track_ids = {}
    for tlinkline in tlinklines:
        if re.search('<TLINK', tlinkline):
            re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+.*\/>'
            m = re.search(re_exp, tlinkline)
            if m:
                id, core, fromtext, ref, totext, relType = m.groups()
            else:
                raise Exception("Malformed EVENT tag: %s" % (tlinkline))
            count+=1
            
            if core!='' and ref!='':
                if len(dic)<3:
                    gold_core=core
                    gold_ref=ref
                else:
                    try:
                        gold_core_tuple=dic[core]
                        gold_core=gold_core_tuple.split('#@#')[0]
                    except KeyError:
                        gold_core=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % core)
                    try:
                        gore_ref_tuple=dic[ref]   
                        gold_ref=gore_ref_tuple.split('#@#')[0]
                    except KeyError:
                        gold_ref=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % ref)
                if gold_core!='' and gold_ref!='' and relType!='':
                    try:
                        gold_core=existing_ids[core]
                    except KeyError:
                        existing_ids[core]=gold_core
                    try:
                        gold_ref=existing_ids[ref]
                    except KeyError:
                        existing_ids[ref]=gold_ref
                    relType=relType.replace('OVERLAP','SIMULTANEOUS')
                    foo = tlink_xml+'\t'+gold_core+'\t'+gold_ref+'\t'+relType+'\n'
                    if debug >= 3: 
                        print(foo) 
                    newtext += foo 
                    track_ids[(core, ref)]=id
    #newtext = change_DURING_relation(newtext)
    return newtext, track_ids

def get_relations_from_dictionary(tlink_xml,dic):
    tlinklines=open(tlink_xml).readlines()
    newtext=''
    existing_ids={}
    count=0
    for tlinkline in tlinklines:
        if re.search('<TLINK', tlinkline):
            re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+.*\/>'
            m = re.search(re_exp, tlinkline)
            if m:
                id, core, fromtext, ref, totext, relType = m.groups()
            else:
                raise Exception("Malformed EVENT tag: %s" % (tlinkline))
            count+=1
            
            if core!='' and ref!='':
                if len(dic)<3:
                    gold_core=core
                    gold_ref=ref
                else:
                    try:
                        gold_core_tuple=dic[core]
                        gold_core=gold_core_tuple.split('#@#')[0]
                    except KeyError:
                        gold_core=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % core)
                    try:
                        gore_ref_tuple=dic[ref]   
                        gold_ref=gore_ref_tuple.split('#@#')[0]
                    except KeyError:
                        gold_ref=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % ref)
                if gold_core!='' and gold_ref!='' and relType!='':
                    try:
                        gold_core=existing_ids[core]
                    except KeyError:
                        existing_ids[core]=gold_core
                    try:
                        gold_ref=existing_ids[ref]
                    except KeyError:
                        existing_ids[ref]=gold_ref
                    relType=relType.replace('OVERLAP','SIMULTANEOUS')
                    foo = tlink_xml+'\t'+gold_core+'\t'+gold_ref+'\t'+relType+'\n'
                    if debug >= 3: 
                        print(foo) 
                    newtext += foo 
    #newtext = change_DURING_relation(newtext)
    return newtext



def reverse_relation(rel): 
    rel = re.sub('"', '', rel) 
    if rel.upper() == 'BEFORE': 
        return 'AFTER'
    if rel.upper() == 'AFTER': 
        return 'BEFORE' 
    if rel.upper() == 'IBEFORE': 
        return 'IAFTER' 
    if rel.upper() == 'IAFTER': 
        return 'IBEFORE' 
    if rel.upper() == 'DURING': 
        return 'DURING_BY' 
    if rel.upper() == 'BEGINS': 
        return 'BEGUN_BY' 
    if rel.upper() == 'BEGUN_BY': 
        return 'BEGINS'
    if rel.upper() == 'ENDS': 
        return 'ENDED_BY' 
    if rel.upper() == 'ENDED_BY': 
        return 'ENDS' 
    if rel.upper() == 'INCLUDES': 
        return 'IS_INCLUDED' 
    if rel.upper() == 'IS_INCLUDED': 
        return 'INCLUDES' 
    return rel.upper() 


def get_triples(tlink_file): 
    tlinks = tlink_file # open(tlink_file).read() # tlink_file # 
    relations = '' 
    for line in tlinks.split('\n'): 
        if line.strip() == '': 
            continue 
        words = line.split('\t') 
        relations += words[0]+'\t'+words[1]+'\t'+words[2]+'\t'+words[3]+'\n'
        relations += words[0]+'\t'+words[2]+'\t'+words[1]+'\t'+reverse_relation(words[3]) +'\n'        
    return relations 


"""
# TODO: adapt this later for global inference, passing in weight_sorted_relations
"""    
def get_timegraphs(gold, system): 
    gold_text = gold # open(gold).read() # gold #
    system_text = system # open(system).read() # system # 

    tg_gold = Timegraph() 
    tg_gold = create_timegraph_from_weight_sorted_relations(gold_text, tg_gold) 
    tg_gold.final_relations = tg_gold.final_relations + tg_gold.violated_relations
    tg_system = Timegraph() 
    tg_system = create_timegraph_from_weight_sorted_relations(system_text, tg_system) 
    tg_system.final_relations = tg_system.final_relations + tg_system.violated_relations
    
    # TODO: interpretation: https://github.com/naushadzaman/tempeval3_toolkit/blob/master/evaluation-relations/relation_to_timegraph.py
    # tg.final_relations = tg.nonredundant - tg.remove_from_reduce - tg.violated_relations 
    return tg_gold, tg_system 

 
 
# extract entities and relation from tlink line 
def get_x_y_rel(tlinks): 
    words = tlinks.split('\t')
    x = words[1]
    y = words[2]
    rel = words[3]
    return x, y, rel 

def get_entity_rel(tlink): 
    words = tlink.split('\t') 
    if len(words) == 3: 
        return words[0]+'\t'+words[1]+'\t'+words[2] 
    return words[1]+'\t'+words[2]+'\t'+words[3] 

def total_relation_matched(A_tlinks, B_tlinks, B_relations, B_tg): 
    """
    Precision = 
        (# of system temporal relations that can be
            verified from reference annotation temporal closure
            graph / # of temporal relations in system output)
    NOTE: 
    
    tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold = A_tlinks, B_tlinks, B_relations, B_tg
    
    A_tlinks = tg_system.final_relations #NOTE: # of system temporal relations
    B_relations = tripled gold_annotations
    B_tg = tg_gold
    """
    
    count = 0 
    for tlink in A_tlinks.split('\n'): 
        if tlink.strip() == '': 
            continue 
        if debug >= 2: 
            print(tlink)
        x, y, rel = get_x_y_rel(tlink) 
        foo = interval_rel_X_Y(x, y, B_tg, rel, 'evaluation') # this is check in closure
        if re.search(get_entity_rel(tlink.strip()), B_relations): 
            count += 1 
            if debug >= 2: 
                print('True') 
            continue 
        if debug >= 2: 
            print(x, y, rel, foo[1])
        if re.search('true', foo[1]):
            count += 1 
    return count 
           
def total_implicit_matched(system_reduced, gold_reduced, gold_tg): 
    count = 0 
    for tlink in system_reduced.split('\n'): 
        if tlink.strip() == '': 
            continue 
        if debug >= 2: 
            print(tlink)
        if re.search(tlink, gold_reduced): 
            continue 

        x, y, rel = get_x_y_rel(tlink) 
        foo = interval_rel_X_Y(x, y, gold_tg, rel, 'evaluation')
        if debug >= 2: 
            print(x, y, rel, foo[1])
        if re.search('true', foo[1]):
            count += 1 
    return count 
    
 
def get_entities(relations): 
    included = '' 
    for each in relations.split('\n'): 
        if each.strip() == '': 
            continue 
        words = each.split('\t')
        if not re.search('#'+words[1]+'#', included):
            included += '#'+words[1]+'#\n'
        if not re.search('#'+words[2]+'#', included):
            included += '#'+words[2]+'#\n'
    return included

def get_n(relations): 
    included = get_entities(relations) 
    return (len(included.split('\n'))-1)

def get_common_n(gold_relations, system_relations): 
    gold_entities = get_entities(gold_relations) 
    system_entities = get_entities(system_relations) 
    common = '' 
    for each in gold_entities.split('\n'): 
        if each.strip() == '':
            continue 
        if re.search(each, system_entities): 
            common += each + '\n' 
    if debug >= 3: 
        print(len(gold_entities.split('\n')), len(system_entities.split('\n')), len(common.split('\n'))) 
        print(common.split('\n'))
        print(gold_entities.split('\n'))
    return (len(common.split('\n'))-1)

def get_ref_minus(gold_relation, system_relations): 
    system_entities = get_entities(system_relations)
    count = 0 
    for each in gold_relation.split('\n'): 
        if each.strip() == '': 
            continue 
        words = each.split('\t')
        if re.search('#'+words[1]+'#', system_entities) and re.search('#'+words[2]+'#', system_entities):
            count += 1 
    return count 




def evaluate_two_files_for_i2b2(arg1, arg2,dic,dicsys):
    
    # global global_prec_matched
    # global global_rec_matched
    # global global_system_total
    # global global_gold_total

    if debug >= 1: 
        print('\n\n Evaluate', arg1, arg2)
        
    """
    # DATA: 
    gold_annotations:
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E136	E135	BEFORE
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml\tE84\tE98\tSIMULTANEOUS\n

    system_annotations
    i2b2_models/relations/ner/ner_bert-base-uncased_0_2e-5_10_32_1/predicted_xml/101.xml	E125	T1	SIMULTANEOUS
    """
    gold_annotation= get_relations(arg1,dicsys,dic)
    system_annotation = get_relations_from_dictionary(arg2,dic) 

    """
    two_times_graphs
    
    """
    tg_gold, tg_system = get_timegraphs(gold_annotation, system_annotation) 
    
    # data: 
    """ gold_relations, system_relations
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E84	E98	SIMULTANEOUS
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E98	E84	SIMULTANEOUS    
    for each link flip it
    """
    gold_relations = get_triples(gold_annotation) 
    system_relations = get_triples(system_annotation) 
    
    # NOTE: for precision
    gold_count=len(tg_gold.final_relations.split('\n'))-1
    sys_count=len(tg_system.final_relations.split('\n'))-1
    if debug >= 2: 
        print('\nchecking precision') 
    """
    Precision = 
        (# of system temporal relations that can be
            verified from reference annotation temporal closure
            graph / # of temporal relations in system output)
    """
    """
    #NOTE: (tg_gold.final_relations) is # of temporal relations in system output, not used in function
    - tg_system.final_relations is # of system temporal relations
    - gold_relations: tripled gold_annotations
    - tg_gold: get_timegraphs(gold_annotation, system_annotation) 
    """
    # this is for the whole_data_set
    prec_matched = total_relation_matched(tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold) 
    # TODO: for every subset:
    #rel_types = ['BEFORE', 'AFTER', 'SIMULTANEOUS']



    # TODO
    #NOTE: for recall 
    if debug >= 2: 
        print('\nchecking recall') 
    """
    Recall = 
        (# of reference annotation temporal relations
            that can be verified from system output’s temporal
            closure graph / # of temporal relations in reference
            annotation) 
    """
    rec_matched = total_relation_matched(tg_gold.final_relations, tg_system.final_relations, system_relations, tg_system) 
    #     #ground_truth_output_subgroup = {}
    # def get_rel_dict_by_subgroup(system_final_relations):
    #     system_output_subgroup = {t:'' for t in rel_types}
    #     system_count_subgorup = {t:0 for t in rel_types}
    #     for rel_value in system_final_relations.split('\n'):
    #         if rel_value!='':
    #             _, e1, e2, rel, _ = rel_value.split("\t")
    #             system_output_subgroup[rel] += '\t'+e1+'\t'+e2+'\t'+rel+'\t\n' #"\n".join([ for line in tg_system.final_relations.split('\n')])
    #             system_count_subgorup[rel] += 1
    #     return system_output_subgroup, system_count_subgorup
    # system_output_subgroup, system_count_subgroup = get_rel_dict_by_subgroup(tg_system.final_relations)
    # gold_output_subgroup, gold_count_subgroup = get_rel_dict_by_subgroup(tg_gold.final_relations)
    
    # subgroup_dict = {}
    # for rtype in rel_types:
    #     sub_prec_matched = total_relation_matched(system_output_subgroup[rtype], tg_gold.final_relations, gold_relations, tg_gold) 
    #     sub_rec_matched = total_relation_matched(gold_output_subgroup[rtype], tg_system.final_relations, system_relations, tg_system) 
    #     sub_sys_count = system_count_subgroup[rtype]
    #     sub_gold_count = gold_count_subgroup[rtype]
    #     subgroup_dict[rtype] = [sub_sys_count, sub_gold_count, sub_prec_matched, sub_rec_matched]
    
    return sys_count, len(tg_gold.final_relations.split('\n'))-1, prec_matched, rec_matched #subgroup_dict


# TODO: do this for "BEFORE", "AFTER", SIMULTANEOUS
def evaluate_two_files_for_subgroup_analysis(arg1, arg2,dic,dicsys):
    
    # global global_prec_matched
    # global global_rec_matched
    # global global_system_total
    # global global_gold_total

    if debug >= 1: 
        print('\n\n Evaluate', arg1, arg2)
        
    """
    # DATA: 
    gold_annotations:
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E136	E135	BEFORE
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml\tE84\tE98\tSIMULTANEOUS\n

    system_annotations
    i2b2_models/relations/ner/ner_bert-base-uncased_0_2e-5_10_32_1/predicted_xml/101.xml	E125	T1	SIMULTANEOUS
    """
    gold_annotation= get_relations(arg1,dicsys,dic)
    system_annotation = get_relations_from_dictionary(arg2,dic) 

    """
    two_times_graphs
    
    """
    tg_gold, tg_system = get_timegraphs(gold_annotation, system_annotation) 
    
    # data: 
    """ gold_relations, system_relations
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E84	E98	SIMULTANEOUS
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E98	E84	SIMULTANEOUS    
    for each link flip it
    """
    gold_relations = get_triples(gold_annotation) 
    system_relations = get_triples(system_annotation) 
    
    # NOTE: for precision
    gold_count=len(tg_gold.final_relations.split('\n'))-1
    sys_count=len(tg_system.final_relations.split('\n'))-1
    if debug >= 2: 
        print('\nchecking precision') 
    """
    Precision = 
        (# of system temporal relations that can be
            verified from reference annotation temporal closure
            graph / # of temporal relations in system output)
    """
    """
    #NOTE: (tg_gold.final_relations) is # of temporal relations in system output, not used in function
    - tg_system.final_relations is # of system temporal relations
    - gold_relations: tripled gold_annotations
    - tg_gold: get_timegraphs(gold_annotation, system_annotation) 
    """
    # this is for the whole_data_set
    prec_matched = total_relation_matched(tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold) 
    # TODO: for every subset:
    rel_types = ['BEFORE', 'AFTER', 'SIMULTANEOUS']



    # TODO
    #NOTE: for recall 
    if debug >= 2: 
        print('\nchecking recall') 
    """
    Recall = 
        (# of reference annotation temporal relations
            that can be verified from system output’s temporal
            closure graph / # of temporal relations in reference
            annotation) 
    """
    rec_matched = total_relation_matched(tg_gold.final_relations, tg_system.final_relations, system_relations, tg_system) 
        #ground_truth_output_subgroup = {}
    def get_rel_dict_by_subgroup(system_final_relations):
        system_output_subgroup = {t:'' for t in rel_types}
        system_count_subgorup = {t:0 for t in rel_types}
        for rel_value in system_final_relations.split('\n'):
            if rel_value!='':
                _, e1, e2, rel, _ = rel_value.split("\t")
                system_output_subgroup[rel] += '\t'+e1+'\t'+e2+'\t'+rel+'\t\n' #"\n".join([ for line in tg_system.final_relations.split('\n')])
                system_count_subgorup[rel] += 1
        return system_output_subgroup, system_count_subgorup
    system_output_subgroup, system_count_subgroup = get_rel_dict_by_subgroup(tg_system.final_relations)
    gold_output_subgroup, gold_count_subgroup = get_rel_dict_by_subgroup(tg_gold.final_relations)
    
    subgroup_dict = {}
    for rtype in rel_types:
        sub_prec_matched = total_relation_matched(system_output_subgroup[rtype], tg_gold.final_relations, gold_relations, tg_gold) 
        sub_rec_matched = total_relation_matched(gold_output_subgroup[rtype], tg_system.final_relations, system_relations, tg_system) 
        sub_sys_count = system_count_subgroup[rtype]
        sub_gold_count = gold_count_subgroup[rtype]
        subgroup_dict[rtype] = [sub_sys_count, sub_gold_count, sub_prec_matched, sub_rec_matched]
    
    return sys_count, len(tg_gold.final_relations.split('\n'))-1, prec_matched, rec_matched, subgroup_dict



def evaluate_two_files_for_link_type_based_analysis(arg1, arg2,dic,dicsys):
    """
        E-T: 1. sectime 2. other E-T
             3. T-T
             4. E-E
    """
    # global global_prec_matched
    # global global_rec_matched
    # global global_system_total
    # global global_gold_total

    if debug >= 1: 
        print('\n\n Evaluate', arg1, arg2)
        
    """
    # DATA: 
    gold_annotations:
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E136	E135	BEFORE
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml\tE84\tE98\tSIMULTANEOUS\n

    system_annotations
    i2b2_models/relations/ner/ner_bert-base-uncased_0_2e-5_10_32_1/predicted_xml/101.xml	E125	T1	SIMULTANEOUS
    """
    # gold_annotation= get_relations(arg1,dicsys,dic)
    # system_annotation = get_relations_from_dictionary(arg2,dic) 

    gold_annotation, gold_link_ids = get_relations_plus_ids(arg1,dicsys,dic)
    system_annotation, system_link_ids = get_relations_from_dictionary_plus_ids(arg2,dic) 
    """
    two_times_graphs
    
    """
    tg_gold, tg_system = get_timegraphs(gold_annotation, system_annotation) 
    
    # data: 
    """ gold_relations, system_relations
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E84	E98	SIMULTANEOUS
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E98	E84	SIMULTANEOUS    
    for each link flip it
    """
    gold_relations = get_triples(gold_annotation) 
    system_relations = get_triples(system_annotation) 
    
    # NOTE: for precision
    gold_count=len(tg_gold.final_relations.split('\n'))-1
    sys_count=len(tg_system.final_relations.split('\n'))-1
    if debug >= 2: 
        print('\nchecking precision') 
    """
    Precision = 
        (# of system temporal relations that can be
            verified from reference annotation temporal closure
            graph / # of temporal relations in system output)
    """
    """
    #NOTE: (tg_gold.final_relations) is # of temporal relations in system output, not used in function
    - tg_system.final_relations is # of system temporal relations
    - gold_relations: tripled gold_annotations
    - tg_gold: get_timegraphs(gold_annotation, system_annotation) 
    """
    # this is for the whole_data_set
    prec_matched = total_relation_matched(tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold) 
    # TODO: for every subset:
    
    """
        E-T: 1. sectime 2. other E-T
             3. T-T
             4. E-E
    """
    #rel_types = ['BEFORE', 'AFTER', 'SIMULTANEOUS']
    # other is other ET
    rel_types = ['SEC', 'ET_OTHER', 'ET', 'TT', 'EE']


    # TODO
    #NOTE: for recall 
    if debug >= 2: 
        print('\nchecking recall') 
    """
    Recall = 
        (# of reference annotation temporal relations
            that can be verified from system output’s temporal
            closure graph / # of temporal relations in reference
            annotation) 
    """
    rec_matched = total_relation_matched(tg_gold.final_relations, tg_system.final_relations, system_relations, tg_system) 
        #ground_truth_output_subgroup = {}
    def get_rel_dict_by_type(system_final_relations):
        system_output_type = {t:'' for t in rel_types}
        system_count_type = {t:0 for t in rel_types}
        for rel_value in system_final_relations.split('\n'):
            if rel_value!='':
                _, e1, e2, rel, _ = rel_value.split("\t")
                curr_output = '\t'+e1+'\t'+e2+'\t'+rel+'\t\n' #"\n".join([ for line in tg_system.final_relations.split('\n')])
                # SECTIME
                if "sec" in system_link_ids[(e1, e2)].lower() or ((e2, e1) in system_link_ids and "sec" in system_link_ids[(e2, e1)].lower()): #or "sec" in system_link_ids[(e2, e1)].lower(): # DEBUG: do I need to add E2, E1?
                    
                    lType = 'SEC'
                    system_output_type[lType] += curr_output
                    system_count_type[lType] += 1
                    system_output_type['ET'] += curr_output
                    system_count_type['ET'] += 1
                # elif (e2, e1) in system_link_ids:
                #     if "sec" in system_link_ids[(e2, e1)].lower():
                #         print("need to rerun all ") 
                else:
                    r_e_type = r"(e)(\d+)"
                    if (event_is_e(r_e_type, e1) and not event_is_e(r_e_type, e2)) or (not event_is_e(r_e_type, e1) and event_is_e(r_e_type, e2)):
                        lType = 'ET_OTHER'
                        system_output_type[lType] += curr_output
                        system_count_type[lType] += 1
                        system_output_type['ET'] += curr_output
                        system_count_type['ET'] += 1
                    elif not event_is_e(r_e_type, e1) and not event_is_e(r_e_type, e2):
                        lType = 'TT'
                        system_output_type[lType] += curr_output
                        system_count_type[lType] += 1
                    else: # ee
                        lType = 'EE'
                        system_output_type[lType] += curr_output
                        system_count_type[lType] += 1
        return system_output_type, system_count_type
    system_output_type, system_count_type = get_rel_dict_by_type(tg_system.final_relations)
    gold_output_subgroup, gold_count_subgroup = get_rel_dict_by_type(tg_gold.final_relations)
    
    subgroup_dict = {}
    for rtype in rel_types:
        sub_prec_matched = total_relation_matched(system_output_type[rtype], tg_gold.final_relations, gold_relations, tg_gold) 
        sub_rec_matched = total_relation_matched(gold_output_subgroup[rtype], tg_system.final_relations, system_relations, tg_system) 
        sub_sys_count = system_count_type[rtype]
        sub_gold_count = gold_count_subgroup[rtype]
        subgroup_dict[rtype] = [sub_sys_count, sub_gold_count, sub_prec_matched, sub_rec_matched]
    
    return sys_count, len(tg_gold.final_relations.split('\n'))-1, prec_matched, rec_matched, subgroup_dict




def evaluate_two_files_for_marker_type_based_analysis(arg1, arg2,dic,dicsys, doc_id=None, marker_dict=None, rel_types=None):
    """
        E-T: 1. sectime 2. other E-T
             3. T-T
             4. E-E
    """
    # global global_prec_matched
    # global global_rec_matched
    # global global_system_total
    # global global_gold_total

    if debug >= 1: 
        print('\n\n Evaluate', arg1, arg2)
        
    """
    # DATA: 
    gold_annotations:
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E136	E135	BEFORE
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml\tE84\tE98\tSIMULTANEOUS\n

    system_annotations
    i2b2_models/relations/ner/ner_bert-base-uncased_0_2e-5_10_32_1/predicted_xml/101.xml	E125	T1	SIMULTANEOUS
    """
    # gold_annotation= get_relations(arg1,dicsys,dic)
    # system_annotation = get_relations_from_dictionary(arg2,dic) 

    gold_annotation, gold_link_ids = get_relations_plus_ids(arg1,dicsys,dic)
    system_annotation, system_link_ids = get_relations_from_dictionary_plus_ids(arg2,dic) 
    """
    two_times_graphs
    
    """
    tg_gold, tg_system = get_timegraphs(gold_annotation, system_annotation) 
    
    # data: 
    """ gold_relations, system_relations
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E84	E98	SIMULTANEOUS
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E98	E84	SIMULTANEOUS    
    for each link flip it
    """
    gold_relations = get_triples(gold_annotation) 
    system_relations = get_triples(system_annotation) 
    
    # NOTE: for precision
    gold_count=len(tg_gold.final_relations.split('\n'))-1
    sys_count=len(tg_system.final_relations.split('\n'))-1
    if debug >= 2: 
        print('\nchecking precision') 
    """
    Precision = 
        (# of system temporal relations that can be
            verified from reference annotation temporal closure
            graph / # of temporal relations in system output)
    """
    """
    #NOTE: (tg_gold.final_relations) is # of temporal relations in system output, not used in function
    - tg_system.final_relations is # of system temporal relations
    - gold_relations: tripled gold_annotations
    - tg_gold: get_timegraphs(gold_annotation, system_annotation) 
    """
    # this is for the whole_data_set
    prec_matched = total_relation_matched(tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold) 
    # TODO: for every subset:
    
    """
        E-T: 1. sectime 2. other E-T
             3. T-T
             4. E-E
    """
    #rel_types = ['BEFORE', 'AFTER', 'SIMULTANEOUS']
    # other is other ET
    #rel_types = ['SEC', 'ET_OTHER', 'ET', 'TT', 'EE']


    # TODO
    #NOTE: for recall 
    if debug >= 2: 
        print('\nchecking recall') 
    """
    Recall = 
        (# of reference annotation temporal relations
            that can be verified from system output’s temporal
            closure graph / # of temporal relations in reference
            annotation) 
    """
    rec_matched = total_relation_matched(tg_gold.final_relations, tg_system.final_relations, system_relations, tg_system) 
        #ground_truth_output_subgroup = {}
        
        
    # DEBUG: no swapped order in tg_ystem.final_relations, always in system_link_ids    
    # system_final_relations = tg_system.final_relations
    # count_in_ids = 0
    # exists_flipped_in_final_relations = False
    # count_pair = {}
    # for rel_value in system_final_relations.split('\n'):
    #     if rel_value!='':
    #         _, e1, e2, rel, _ = rel_value.split("\t")
    #         if (e1, e2) in system_link_ids:
    #             count_in_ids += 1
                
    #         if (e2, e1) in count_pair:
    #             count_pair[e2, e1] += 1
    #         else:
    #             if (e1, e2) not in count_pair:
    #                 count_pair[e1, e2] = 1
    #             else:
    #                 print("should not hit here")
    #     for k, v in count_pair.items():
    #         if v > 1:
    #             print("swapped order exists!!!!!!!") 
    #             import pdb
    #             pdb.set_trace()           
    
                
    #assert count_in_ids==(len(tg_system.final_relations)-1) # 248, 5135
    # DEBUG    
        
        
    def get_rel_dict_by_type(system_final_relations):
        system_output_type = {t:'' for t in rel_types}
        system_count_type = {t:0 for t in rel_types}
        for rel_value in system_final_relations.split('\n'):
            if rel_value!='':
                _, e1, e2, rel, _ = rel_value.split("\t")
                #lid = system_link_ids[(e1, e2)]
                #e1 
                
                if e1 in ['Discharge', 'Admission']:
                    marker_dict[doc_id][e1] = "ADMISSION" if e1=='Admission' else "DISCHARGE"
                if e2 in ['Discharge', 'Admission']:
                    marker_dict[doc_id][e2] = "ADMISSION" if e2=='Admission' else "DISCHARGE"
                    
                pair_type = (marker_dict[doc_id][e1], marker_dict[doc_id][e2])
                curr_output = '\t'+e1+'\t'+e2+'\t'+rel+'\t\n' #"\n".join([ for line in tg_system.final_relations.split('\n')])
                if tuple(pair_type) in system_output_type:
                    system_output_type[tuple(pair_type)] += curr_output
                if tuple(pair_type) in system_count_type:
                    system_count_type[tuple(pair_type)] += 1
                # # SECTIME
                # if "sec" in system_link_ids[(e1, e2)].lower():
                #     lType = 'SEC'
                #     system_output_type[lType] += curr_output
                #     system_count_type[lType] += 1
                #     system_output_type['ET'] += curr_output
                #     system_count_type['ET'] += 1
                # else:
                #     r_e_type = r"(e)(\d+)"
                #     if (event_is_e(r_e_type, e1) and not event_is_e(r_e_type, e2)) or (not event_is_e(r_e_type, e1) and event_is_e(r_e_type, e2)):
                #         lType = 'ET_OTHER'
                #         system_output_type[lType] += curr_output
                #         system_count_type[lType] += 1
                #         system_output_type['ET'] += curr_output
                #         system_count_type['ET'] += 1
                #     elif not event_is_e(r_e_type, e1) and not event_is_e(r_e_type, e2):
                #         lType = 'TT'
                #         system_output_type[lType] += curr_output
                #         system_count_type[lType] += 1
                #     else: # ee
                #         lType = 'EE'
                #         system_output_type[lType] += curr_output
                #         system_count_type[lType] += 1
        return system_output_type, system_count_type
    system_output_type, system_count_type = get_rel_dict_by_type(tg_system.final_relations)
    gold_output_subgroup, gold_count_subgroup = get_rel_dict_by_type(tg_gold.final_relations)
    
    subgroup_dict = {}
    for rtype in rel_types:
        sub_prec_matched = total_relation_matched(system_output_type[rtype], tg_gold.final_relations, gold_relations, tg_gold) 
        sub_rec_matched = total_relation_matched(gold_output_subgroup[rtype], tg_system.final_relations, system_relations, tg_system) 
        sub_sys_count = system_count_type[rtype]
        sub_gold_count = gold_count_subgroup[rtype]
        subgroup_dict[rtype] = [sub_sys_count, sub_gold_count, sub_prec_matched, sub_rec_matched]
    
    return sys_count, len(tg_gold.final_relations.split('\n'))-1, prec_matched, rec_matched, subgroup_dict


def evaluate_two_files(arg1, arg2,dic,dicsys):

    global global_prec_matched
    global global_rec_matched
    global global_system_total
    global global_gold_total

    if debug >= 1: 
        print('\n\n Evaluate', arg1, arg2)
        
    """
    gold_annotations:
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml	E136	E135	BEFORE
    preprocess/corpus/i2b2/ground_truth/merged_xml/101.xml\tE84\tE98\tSIMULTANEOUS\n

    system_annotations
    i2b2_models/relations/ner/ner_bert-base-uncased_0_2e-5_10_32_1/predicted_xml/101.xml	E125	T1	SIMULTANEOUS
    """
    gold_annotation= get_relations(arg1,dicsys,dic)
    system_annotation = get_relations_from_dictionary(arg2,dic) 

    """
    two_times_graphs
    
    """
    tg_gold, tg_system = get_timegraphs(gold_annotation, system_annotation) 
    

    gold_relations = get_triples(gold_annotation) 
    system_relations = get_triples(system_annotation) 
    #for precision
    gold_count=len(tg_gold.final_relations.split('\n'))-1
    sys_count=len(tg_system.final_relations.split('\n'))-1
    if debug >= 2: 
        print('\nchecking precision') 
    prec_matched = total_relation_matched(tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold) 
    # for recall 
    if debug >= 2: 
        print('\nchecking recall') 
    rec_matched = total_relation_matched(tg_gold.final_relations, tg_system.final_relations, system_relations, tg_system) 
    rec_implicit_matched = total_implicit_matched(tg_system.final_relations, tg_gold.final_relations, tg_gold) 
    n = get_common_n(tg_gold.final_relations, tg_system.final_relations) 
    
##    n = get_n(tg_gold.final_relations)
    ref_plus = 0.5*n*(n-1)
##    ref_minus = len(tg_gold.final_relations.split('\n'))-1
    ref_minus = rec_matched ## get_ref_minus(tg_gold.final_relations, tg_system.final_relations) 
    w = 0.99/(1+ref_plus-ref_minus) # ref_minus #
    if debug >= 2: 
        print('n =', n) 
        print('rec_implicit_matched', rec_implicit_matched)
        print('n, ref_plus, ref_minus', n , ref_plus , ref_minus)
        print('w', w)
        print('rec_matched', rec_matched)
        print('total', gold_count)

        print('w*rec_implicit_matched', w*rec_implicit_matched)

    if debug >= 2: 
        print('precision', prec_matched, sys_count)
    if gold_count <= 0: 
        precision = 0 
    else: 
        precision = prec_matched*1.0/gold_count

    if debug >= 2: 
        print('recall', rec_matched, len(tg_gold.final_relations.split('\n'))-1)
    if len(tg_gold.final_relations.split('\n')) <= 1: 
        recall = 0 
    else:
        recall2 = (rec_matched)*1.0/gold_count
        recall = (rec_matched+w*rec_implicit_matched)*1.0/gold_count
        if debug >= 2: 
            print('recall2', recall2)
            print('recall', recall)
    
    if debug >= 1: 
        print(precision, recall, get_fscore(precision, recall)) 
    global_prec_matched += prec_matched
    global_rec_matched += rec_matched+w*rec_implicit_matched
    global_system_total += sys_count 
    global_gold_total += len(tg_gold.final_relations.split('\n'))-1

    #return tg_system
    return sys_count, len(tg_gold.final_relations.split('\n'))-1, prec_matched, rec_matched


"""
count_relation = 0 
count_node = 0 
count_chains = 0 
count_time = 0 
"""

def get_fscore(p, r): 
    if p+r == 0: 
        return 0 
    return 2.0*p*r/(p+r) 


def final_score(): 
    global global_prec_matched
    global global_rec_matched
    global global_system_total
    global global_gold_total 

    if global_system_total == 0: 
        precision = 0 
    else: 
        precision = global_prec_matched*1.0/global_system_total
    if global_gold_total == 0: 
        recall = 0
    else: 
        recall = global_rec_matched*1.0/global_gold_total
    
    if precision == 0 and recall == 0: 
        fscore = 0 
    else: 
        fscore = get_fscore(precision, recall) 
    print('Overall\tP\tR\tF1') 
    print('\t'+str(100*round(precision, 6))+'\t'+str(100*round(recall, 6))+'\t'+str(100*round(fscore, 6)))

