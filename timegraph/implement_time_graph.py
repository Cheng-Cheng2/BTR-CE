import os 

from eventEvaluation import eventEvaluation
from timexEvaluation import timexEvaluation
from temporal_evaluation_adapted import evaluate_two_files
import re
import os
import argparse
from tlinkEvaluation import tlinkEvaluation

# for one file 
systemDic, goldDic = {}, {}
goldDic=
tlinkScores = evaluate_two_files(os.path.join(goldDir,file),os.path.join(systemDir,file),systemDic,goldDic)




#tlinkScores=evaluate_two_files(os.path.join(goldDir,file),os.path.join(systemDir,file),systemDic,goldDic)

# debug = 0
# def evaluate_two_files(arg1, arg2,dic,dicsys):
    
#     global global_prec_matched
#     global global_rec_matched
#     global global_system_total
#     global global_gold_total

#     if debug >= 1: 
#         print(( '\n\n Evaluate', arg1, arg2))
#     gold_annotation= get_relations(arg1,dicsys,dic)
#     system_annotation = get_relations_from_dictionary(arg2,dic) 
    
#     tg_gold, tg_system = get_timegraphs(gold_annotation, system_annotation) 
#     gold_relations = get_triples(gold_annotation) 
#     system_relations = get_triples(system_annotation) 
#     #for precision
#     gold_count=len(tg_gold.final_relations.split('\n'))-1
#     sys_count=len(tg_system.final_relations.split('\n'))-1
#     if debug >= 2: 
#         print( '\nchecking precision' )
#     prec_matched = total_relation_matched(tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold) 
#     # for recall 
#     if debug >= 2: 
#         print( '\nchecking recall' )
#     rec_matched = total_relation_matched(tg_gold.final_relations, tg_system.final_relations, system_relations, tg_system) 
#     rec_implicit_matched = total_implicit_matched(tg_system.final_relations, tg_gold.final_relations, tg_gold) 
#     n = get_common_n(tg_gold.final_relations, tg_system.final_relations) 
    
# ##    n = get_n(tg_gold.final_relations)
#     ref_plus = 0.5*n*(n-1)
# ##    ref_minus = len(tg_gold.final_relations.split('\n'))-1
#     ref_minus = rec_matched ## get_ref_minus(tg_gold.final_relations, tg_system.final_relations) 
#     w = 0.99/(1+ref_plus-ref_minus) # ref_minus #
#     if debug >= 2: 
#         print(( 'n =', n ))
#         print(( 'rec_implicit_matched', rec_implicit_matched))
#         print(( 'n, ref_plus, ref_minus', n , ref_plus , ref_minus))
#         print(( 'w', w))
#         print(( 'rec_matched', rec_matched))
#         print(( 'total', gold_count))

#         print(( 'w*rec_implicit_matched', w*rec_implicit_matched))

#     if debug >= 2: 
#         print(( 'precision', prec_matched, sys_count))
#     if gold_count <= 0: 
#         precision = 0 
#     else: 
#         precision = prec_matched*1.0/gold_count

#     if debug >= 2: 
#         print(( 'recall', rec_matched, len(tg_gold.final_relations.split('\n'))-1))
#     if len(tg_gold.final_relations.split('\n')) <= 1: 
#         recall = 0 
#     else:
#         recall2 = (rec_matched)*1.0/gold_count
#         recall = (rec_matched+w*rec_implicit_matched)*1.0/gold_count
#         if debug >= 2: 
#             print(( 'recall2', recall2))
#             print(( 'recall', recall))
    
#     if debug >= 1: 
#         print(( precision, recall, get_fscore(precision, recall) ))
#     global_prec_matched += prec_matched
#     global_rec_matched += rec_matched+w*rec_implicit_matched
#     global_system_total += sys_count 
#     global_gold_total += len(tg_gold.final_relations.split('\n'))-1

#     #return tg_system
#     return sys_count, len(tg_gold.final_relations.split('\n'))-1, prec_matched, rec_matched


# """
# count_relation = 0 
# count_node = 0 
# count_chains = 0 
# count_time = 0 
# """

# def get_fscore(p, r): 
#     if p+r == 0: 
#         return 0 
#     return 2.0*p*r/(p+r) 


# def final_score(): 
#     global global_prec_matched
#     global global_rec_matched
#     global global_system_total
#     global global_gold_total 

#     if global_system_total == 0: 
#         precision = 0 
#     else: 
#         precision = global_prec_matched*1.0/global_system_total
#     if global_gold_total == 0: 
#         recall = 0
#     else: 
#         recall = global_rec_matched*1.0/global_gold_total
    
#     if precision == 0 and recall == 0: 
#         fscore = 0 
#     else: 
#         fscore = get_fscore(precision, recall) 
#     print( 'Overall\tP\tR\tF1' )
#     print(( '\t'+str(100*round(precision, 6))+'\t'+str(100*round(recall, 6))+'\t'+str(100*round(fscore, 6))))

