'''
Created on Jul 6, 2012

@author: Weiyi Sun

usage: i2b2Evaluation.py [-h] [--cc] [--oc] [--oo] [--tempeval] [-event]
                         [-timex] [-tlink] [-all] [-overlap] [-exact]
                         [-partialCredit]
                         gold_file_des system_file_des

Evaluate system output against gold standard.

'''

import sys

if sys.version_info<(2,7):
    print "Error: This evaluation script requires Python 2.7 or higher"
else:
    
    from eventEvaluation import eventEvaluation
    from timexEvaluation import timexEvaluation
    from temporal_evaluation_adapted import evaluate_two_files
    import re
    import os
    import argparse
    from tlinkEvaluation import tlinkEvaluation
    
    if __name__ == '__main__':
    
        usage= "%prog [options] [goldstandard-file] [systemOutput-file]" + __doc__
        parser = argparse.ArgumentParser(description='Evaluate system output against gold standard.')
        parser.add_argument('gold_file_des', type=str, nargs=1,\
                         help='the file or directory of the gold standard xml file/directory')
        parser.add_argument('system_file_des', type=str, nargs=1,\
                         help='the file or directory of the system output xml file/directory')
        
        parser.add_argument('--cc', dest='evaluation_option', action='store_const',\
                          const='cc', default='tempeval', help="""select different types of tlink evaluation: 
                          --oo: Origianl against Orignal:""")
        parser.add_argument('--oc', dest='evaluation_option', action='store_const',\
                          const='oc', default='tempeval', help="""select different types of tlink evaluation: 
                          --oc Original against Closure:""")
        parser.add_argument('--oo', dest='evaluation_option', action='store_const',\
                          const='oo', default='tempeval', help="""select different types of tlink evaluation:--oo: Origianl against Orignal:""")
        parser.add_argument('--tempeval', dest='evaluation_option', action='store_const',\
                          const='tempeval', default='tempeval', help='select different types of tlink evaluation: --tempeval Tempeval3 evaluation method(default)')
        
        parser.add_argument('-event', dest='entities_to_evaluate', action='store_const',\
                          const='event', default='all', help='select which tag to run evaluation on: event, timex, tlink, all(default)')
        parser.add_argument('-timex', dest='entities_to_evaluate', action='store_const',\
                          const='timex', default='all', help='select which tag to run evaluation on: event, timex, tlink, all(default)')
        parser.add_argument('-tlink', dest='entities_to_evaluate', action='store_const',\
                          const='tlink', default='all', help='select which tag to run evaluation on: event, timex, tlink, all(default)')
        parser.add_argument('-all', dest='entities_to_evaluate', action='store_const',\
                          const='all', default='all', help='select which tag to run evaluation on: event, timex, tlink, all(default)')
        
        parser.add_argument('-overlap', dest='span_match_type', action='store_const',\
                          const='overlap', default='overlap', help='choose extent match type: overlap(default), exact, partialCredit(1 for exact match, 0.5 for overlap match)')
        parser.add_argument('-exact', dest='span_match_type', action='store_const',\
                          const='exact', default='overlap', help='choose extent match type: overlap(default), exact, partialCredit(1 for exact match, 0.5 for overlap match)')
        parser.add_argument('-partialCredit', dest='span_match_type', action='store_const',\
                          const='partialCredit', default='overlap', help='choose extent match type: overlap(default), exact, partialCredit(1 for exact match, 0.5 for overlap match)')
        
        args = parser.parse_args()
        
        if len(args.gold_file_des)==1 and len(args.system_file_des)==1:
            eval_type_description="""
Evaluation method:
   Evaluating tags: %s
   Span match method: %s
   TLINK evaluation method: %s""" %\
             (args.entities_to_evaluate.upper(), args.span_match_type,args.evaluation_option)
            print eval_type_description
            goldDir=args.gold_file_des[0]
            systemDir=args.system_file_des[0]
            if os.path.isdir(goldDir+'/') and goldDir[-1]<>'/':
                goldDir+='/'
            if os.path.isdir(systemDir+'/') and systemDir[-1]<>"/":
                systemDir+='/'
            if os.path.isdir(goldDir) and os.path.isdir(systemDir):
                goldFileList=os.listdir(goldDir)
                systemFileList=os.listdir(systemDir)
                if len(goldFileList)==len(systemFileList):
                    #eventGoldCount,eventSysCount,eventPrecMatch,eventRecMatch,eventtype,eventPol,eventMod
                    totaleventScores=[0,0,0,0,0,0,0,0.0,0.0]
                    #timexGoldCount,timexSysCount,timexPrecMatch,timexRecMatch,timextype,timexVal,timeMod
                    totaltimexScores=[0,0,0,0,0,0,0,0.0,0.0]
                    #tlinkGoldCount,tlinkSysCount,tlinkPrecMatch,tlinkRecMatch
                    totaltlinkScores=[0,0,0,0]
                    for file in goldFileList:
                        #print '{:*^46}'.format(' '+file+' ')
                        goldDic={}
                        systemDic={}
                        if args.entities_to_evaluate in ['event', 'all']:
                            eventScores=eventEvaluation(os.path.join(goldDir,file),os.path.join(systemDir,file), args.span_match_type)
                            for i in range(2,11):
                                totaleventScores[i-2]+=eventScores[i]
                            goldDic=eventScores[0]
                            systemDic=eventScores[1]
                        if args.entities_to_evaluate in ['timex', 'all']:
                            timexScores=timexEvaluation(os.path.join(goldDir,file),os.path.join(os.path.join(systemDir,file)),args.span_match_type,goldDic,systemDic)
                            for i in range(2,11):
                                totaltimexScores[i-2]+=timexScores[i]
                            goldDic=timexScores[0]
                            systemDic=timexScores[1]
                        if args.entities_to_evaluate in ['tlink','all']:
                            if args.evaluation_option=='tempeval':

                                tlinkScores=evaluate_two_files(os.path.join(goldDir,file),os.path.join(systemDir,file),systemDic,goldDic)
                            elif args.evaluation_option=='oc':
                                tlinkScores=tlinkEvaluation(os.path.join(goldDir,file),os.path.join(systemDir,file),'OrigVsClosure',goldDic,systemDic)
                            elif args.evaluation_option=='cc':
                                tlinkScores=tlinkEvaluation(os.path.join(goldDir,file),os.path.join(systemDir,file),'ClosureVsClosure',goldDic,systemDic)
                            elif args.evaluation_option=='oo':
                                tlinkScores=tlinkEvaluation(os.path.join(goldDir,file),os.path.join(systemDir,file),'OrigVsOrig',goldDic,systemDic)
                            # precLinkCount, recLinkCount, precMatchCount,  recMatchCount = tlinkScores

                            for i in range(0,4):
                                totaltlinkScores[i]+=tlinkScores[i]
                            if tlinkScores[0]>0:
                                precision=float(tlinkScores[2])/tlinkScores[0]
                            else:
                                precision=0.0
                            if tlinkScores[1]>0:
                                recall=float(tlinkScores[3])/tlinkScores[1]
                            else:
                                recall=0.0   
                            if (tlinkScores[0]+tlinkScores[1])>0:
                                averagePR=(tlinkScores[2]+tlinkScores[3])*1.0/(tlinkScores[0]+tlinkScores[1])
                            else:
                                averagePR=0.0
                            if (precision+recall)>0:
                                fScore=2*(precision*recall)/(precision+recall)
                            else:
                                fScore=0.0
                        
                            '''print """
            Total number of comparable TLINKs: 
               Gold Standard : \t\t"""+str(tlinkScores[1])+"""
               System Output : \t\t"""+str(tlinkScores[0])+"""
            --------------
            Recall : \t\t\t"""+'%.4f'%(recall)+"""
            Precision: \t\t\t""" + '%.4f'%(precision)+"""
            Average P&R : \t\t"""+'%.4f'%(averagePR)+"""
            F measure : \t\t"""+'%.4f'%(fScore)+'\n'
                            '''

                    #calculate accumulated scores
                    if args.entities_to_evaluate in ['event', 'all']:
                        eventGoldCount,eventSysCount,eventPrecMatch,eventRecMatch,eventtype,eventPol,eventMod,eventRecPC,eventPrecPC=totaleventScores
                        if eventGoldCount>0:
                            if args.span_match_type=='partialCredit':
                                final_event_recall=1.0*eventRecPC/eventGoldCount
                            else:
                                final_event_recall=1.0*eventRecMatch/eventGoldCount
                            final_event_type=1.0*eventtype/eventGoldCount
                            final_event_pol=1.0*eventPol/eventGoldCount
                            final_event_mod=1.0*eventMod/eventGoldCount
                        else:
                            final_event_recall=0.0
                            final_event_type=0.0
                            final_event_pol=0.0
                            final_event_mod=0.0
                        if eventSysCount>0:
                            if args.span_match_type=='partialCredit':
                                final_event_prec=1.0*eventPrecPC/eventSysCount
                            else:
                                final_event_prec=1.0*eventPrecMatch/eventSysCount
                        else:
                            final_event_prec=0.0
                        if (eventGoldCount+eventSysCount)>0:
                            averagePR=(final_event_prec*eventSysCount+final_event_recall*eventGoldCount)*1.0/(eventGoldCount+eventSysCount)
                        else:
                            averagePR=0.0
                        if (final_event_recall+final_event_prec)>0:
                            fScore=2*(final_event_prec*final_event_recall)/(final_event_recall+final_event_prec)
                        else:
                            fScore=0.0
                        print '{:*^46}'.format(' Aggregated Scores: ')+"""
            EVENT: 
               Precision : \t"""+'%.4f'%(final_event_prec)+"""
               Recall : \t"""+'%.4f'%(final_event_recall)+"""
               Average P&R : \t"""+'%.4f'%(averagePR)+"""
               F measure : \t"""+'%.4f'%(fScore)+"""
               --------------
               type :\t\t"""+'%.4f'%(final_event_type)+"""
               Polarity :\t"""+'%.4f'%(final_event_pol)+"""
               Modality :\t"""+'%.4f'%(final_event_mod)
                        if args.entities_to_evaluate=='event':
                            print '\n{:*^47}'.format('')
                    if args.entities_to_evaluate in ['timex', 'all']:
                        timexGoldCount,timexSysCount,timexPrecMatch,timexRecMatch,timextype,timexVal,timexMod,tRecPC,tPrecPC=totaltimexScores
                        if timexGoldCount>0:
                            if args.span_match_type=='partialCredit':
                                final_timex_recall=1.0*tRecPC/timexGoldCount
                            else:
                                final_timex_recall=1.0*timexRecMatch/timexGoldCount
                            final_timex_type=1.0*timextype/timexGoldCount
                            final_timex_val=1.0*timexVal/timexGoldCount
                            final_timex_mod=1.0*timexMod/timexGoldCount
                        else:
                            final_timex_recall=0.0
                            final_timex_type=0.0
                            final_timex_val=0.0
                            final_timex_mod=0.0
                        if timexSysCount>0:
                            if args.span_match_type=='partialCredit':
                                final_timex_prec=1.0*tPrecPC/timexSysCount
                            else:
                                final_timex_prec=1.0*timexPrecMatch/timexSysCount
                        else:
                            final_timex_prec=0.0
                        if (timexGoldCount+timexSysCount)>0:
                            averagePR=(final_timex_prec*timexSysCount+final_timex_recall*timexGoldCount)*1.0/(timexGoldCount+timexSysCount)
                        else:
                            averagePR=0.0
                        if (final_timex_recall+final_timex_prec)>0:
                            fScore=2*(final_timex_prec*final_timex_recall)/(final_timex_recall+final_timex_prec)
                        else:
                            fScore=0.0
                        '''
                        if args.entities_to_evaluate=='timex':
                            print '{:*^46}'.format(' Aggregated Scores: ')
                        print """

            TIMEX3:
                Precision : \t"""+'%.4f'%(final_timex_prec)+"""
                Recall : \t"""+'%.4f'%(final_timex_recall)+"""
                Average P&R : \t"""+'%.4f'%(averagePR)+"""
                F measure : \t"""+'%.4f'%(fScore)+"""
                --------------
                type :\t\t"""+'%.4f'%(final_timex_type)+"""
                Val :\t\t"""+'%.4f'%(final_timex_val)+"""
                Modifier :\t"""+'%.4f'%(final_timex_mod)
                        if args.entities_to_evaluate=='timex':
                            print '\n{:*^47}'.format('')'''
                    if args.entities_to_evaluate in ['tlink', 'all']:
                        tlinkSysCount,tlinkGoldCount,tlinkPrecMatch,tlinkRecMatch=totaltlinkScores
                        if tlinkGoldCount>0:
                            final_tlink_recall=1.0*tlinkRecMatch/tlinkGoldCount
                        else:
                            final_tlink_recall=0.0
                        if tlinkSysCount>0:
                            final_tlink_prec=1.0*tlinkPrecMatch/tlinkSysCount
                        else:
                            final_tlink_prec=0.0
                        if (tlinkGoldCount+tlinkSysCount)>0:
                            averagePR=(final_tlink_prec*tlinkSysCount+final_tlink_recall*tlinkGoldCount)*1.0/(tlinkGoldCount+tlinkSysCount)
                        else:
                            averagePR=0.0
                        if (final_tlink_recall+final_tlink_prec)>0:
                            fScore=2*(final_tlink_prec*final_tlink_recall)/(final_tlink_recall+final_tlink_prec)
                        else:
                            fScore=0.0
                        if args.entities_to_evaluate=='tlink':
                            print '{:*^46}'.format(' Aggregated Scores: ')
                        print """
                           
            TLINK:
                Precision : \t"""+'%.4f'%(final_tlink_prec)+"""
                Recall : \t"""+'%.4f'%(final_tlink_recall)+"""
                Average P&R : \t"""+'%.4f'%(averagePR)+"""
                F measure : \t"""+'%.4f'%(fScore)+"""
                """+'\n{:*^47}'.format('')
                            
                        if args.entities_to_evaluate=='tlink':
                            print "WARNING: Running TLINK evaluation by itself assumes gold standard EVENTs/TIMEX3s in both gold standard and system output. If that is not the case, please make sure that the EVENT/TIMEX3 ids in the gold standard match the ids in the system output. Otherwise, the result will be wrong. If in doubt, use -all option."
                else:
                    print "\nError: the files in the gold standard directory and the system directory don't match!\n"
                
                

            elif os.path.isfile(args.gold_file_des[0]) and os.path.isfile(args.system_file_des[0]):
                gold=args.gold_file_des[0]
                system=args.system_file_des[0]
                goldDic={}
                systemDic={}
                if args.entities_to_evaluate in ['event', 'all']:
                    eventScores=eventEvaluation(gold,system,args.span_match_type)
                    goldDic=eventScores[0]
                    systemDic=eventScores[1]
                if args.entities_to_evaluate in ['timex', 'all']:
                    timexScores=timexEvaluation(gold,system,args.span_match_type, goldDic,systemDic)
                    goldDic=timexScores[0]
                    systemDic=timexScores[1]
                if args.entities_to_evaluate in ['tlink', 'all']:
                    if args.evaluation_option=='tempeval':
                        tlinkScores=evaluate_two_files(gold,system,systemDic,goldDic)
                    elif args.evaluation_option=='oc':
                        tlinkScores=tlinkEvaluation(gold,system,'OrigVsClosure',goldDic,systemDic)
                    elif args.evaluation_option=='cc':
                        tlinkScores=tlinkEvaluation(gold,system,'ClosureVsClosure',goldDic,systemDic)
                    elif args.evaluation_option=='oo':
                        tlinkScores=tlinkEvaluation(gold,system,'OrigVsOrig',goldDic,systemDic)
                    # precLinkCount, recLinkCount, precMatchCount,  recMatchCount = tlinkScores
                    if tlinkScores[0]>0:
                        precision=float(tlinkScores[2])/tlinkScores[0]
                    else:
                        precision=0.0
                    if tlinkScores[1]>0:
                        recall=float(tlinkScores[3])/tlinkScores[1]
                    else:
                        recall=0.0   
                    if (tlinkScores[0]+tlinkScores[1])>0:
                        averagePR=(tlinkScores[2]+tlinkScores[3])*1.0/(tlinkScores[0]+tlinkScores[1])
                    else:
                        averagePR=0.0
                    if (precision+recall)>0:
                        fScore=2*(precision*recall)/(precision+recall)
                    else:
                        fScore=0.0
                    
                    print """
            Total number of comparable TLINKs: 
               Gold Standard : \t\t"""+str(tlinkScores[1])+"""
               System Output : \t\t"""+str(tlinkScores[0])+"""
            --------------
            Recall : \t\t\t"""+'%.4f'%(recall)+"""
            Precision: \t\t\t""" + '%.4f'%(precision)+"""
            Average P&R : \t\t"""+'%.4f'%(averagePR)+"""
            F measure : \t\t"""+'%.4f'%(fScore)+'\n'
               
                    if args.entities_to_evaluate=='tlink':
                        print "WARNING: when using tlink evaluation alone, please make sure that the Event/Timex ids in the gold standard matches the ids in the system output. Otherwise, the result will be wrong. If in doubt, use -all option."
            else:
                print "Error: Please input two directorys or two files"
 
