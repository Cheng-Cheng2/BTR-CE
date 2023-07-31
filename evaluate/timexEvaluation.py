'''
Created on Dec 26, 2011

@author: Weiyi Sun


Evaluates system output Timex3s against gold standard Timex3s

- usage:
  $ python timexEvaluation.py goldstandard_xml_filename system_output_xml_filename
  
  - Overlaping extent are considered as matches
  - Recall:
       number of system output that overlaps with gold standard timex extent
  - Precision:
       number of gold standard timex that overlap with system output
  - Attribute score:
       the percentage of correct attributes in the total matched timexes
       e.g. system outputs 5 timexes, 3 of which can be verified in the goldstandard
            2 out of the 3 have the same Timex3 'type' attribute as the goldstandard
            the system type match score will be 2/3=66.6%
       - type, MOD attribute: exact attribute match, normalize for upper/lower case
       - VAL attribute: will normalize for:
       1) missing time designator 'T' in unambiguous cases: 
            e.g. P23H = PT23H, but PT3M<>P3M; 
       2) normalize for upper/lower cases
       3) normalize for using different unit: PT24H==P1D, P8W=P2M
       4) taking into account APPROX|MORE|LESS modifier, 'a few days' with mod='APPROX'
          the val can be assigned as 'P2D' or 'P3D', both are correct. But the quantifier
          should not be off by more than +/-2

'''
import sys

if sys.version_info<(2,7):
    print "Error: This evaluation script requires Python 2.7 or higher"
else:
    import argparse
    import os
    import sys
    import re
    
    
    punctuationsStr= ", . ? ! \" \' < > ; : / \\ ~ _ - + = ( ) [ ] { } | @ # $ % ^ & * ` &apos; &amp; &quot; &gt; &lt;"
    punctuations=punctuationsStr.split()
    
    def open_file(fname):
        if os.path.exists(fname):
            f = open(fname)
            return f
        else:
            outerror("No such file: %s" % fname)
            return None
        
    def list_dir(dirname):
        if os.path.exists(dirname):
            return os.listdir(dirname)
        else:
            outerror("No such file: %s" % dirname)
            return None
        
    def outerror(text):
        #sys.stderr.write(text + "\n")
        raise Exception(text)
    
    def attr_by_line(timex_line):
        """
        Args:
          line - str: MAE TIMEX3 tag line,
                      e.g. <TIMEX3 id="T18" start="3646"
                      end="3652" text="4/2/99" type="DATE" val="1999-04-02" mod="NA" />
        """
        re_exp = 'id=\"([^"]*)\"\s+start=\"([^"]*)\"\s+end=\"([^"]*)\"\s+text=\"([^"]*)\"\s+type=\"([^"]*)\"\s+val=\"([^"]*)\"\s+mod=\"([^"]*)\"\s+\/>'
        m = re.search(re_exp, timex_line)
        if m:
            id, start, end, text, type, val, mod = m.groups()
        else:
            raise Exception("Malformed Timex3 tag: %s" % (timex_line))
        return id, start, end, text, type.upper(), val.upper(), mod.upper()
    
    
    def get_timex(text_fname):
        tf=open_file(text_fname)
        lines = tf.readlines()
        timexes=[]
        # fix for ruling out duplicate ids
        unique_ids=[]
        for line in lines:  
            if re.search('<TIMEX3', line):
                timexTuple=attr_by_line(line)
                if timexTuple[0] not in unique_ids:
                    timexes.append(timexTuple)
                    unique_ids.append(timexTuple[0])
        return timexes
    
    def DurationFrequencyValCompare(val1,mod1,val2,mod2):
        '''
        compares val attribute for Duration / Frequency type Timexes
        
        args:
        val1: val field of the first timex
        mod1: mod field of the first timex
        val2: val field of the second timex
        mod2: mod field of the second timex
        
        Output:
        0: wrong
        1: correct
        '''
        if val1=='' or val2=='':
            return 0
        else:
            if val1[0]=='R': # if val1 is frequency
                if val2[0]<>'R':
                    return 0
                else: # val2 is also frequency
                    if val1.find('P')>-1 and val2.find('P')>-1: # both val1 and val2 are RXPXX
                        repeat1,period1=val1.split('P')
                        repeat2,period2=val2.split('P')
                        if repeat1==repeat2 and comparePeriod(period1,mod1,period2,mod2)==1:
                            return 1
                        else:
                            return 0
                    else: # cases like 'R' or 'R5' with no period information
                        if val1==val2:
                            return 1
                        else:
                            return 0
            else: #if val1 is duration
                if val2[0]=='R':
                    return 0
                else:
                    if val1[0]=='P' and val2[0]=='P':
                        return  comparePeriod(val1[1:],mod1,val2[1:],mod2)
                    else:
                        return 0
                    
            
    def comparePeriod(period1,mod1,period2,mod2):
        '''
        compare whether the Period part of the val is correct
        input: dropped the initial 'P' in duration val, and the 'RXP' in frequency val
        '''
        # comparison for periods in the same unit(Y M W D H S M), 
        # if mod = APPROX|MORE|LESS then allow for +/- 2:
        unit1=period1[-1]
        unit2=period2[-1]
        if unit1==unit2:
            if unit1=='M' and (period1[0]=='T' or period2[0]=='T'): # for minutes, the val field has to contain 'T' to be considered correct
                if period1[0]=='T' and period2[0]=='T':
                    if mod1=='NA' and mod2=='NA':
                        if period1==period2:
                            return 1
                        else:
                            return 0
                    else:
                        quant1=period1[1:-1]
                        quant2=period2[1:-1]
                        if abs(float(quant1)-float(quant2))<3:
                            return 1
                        else:
                            return 0
                else:
                    return 0
            else: # for hours, and secords, allow for the annotation mistake for leaving out the 'T'
                if period2[0]=='T':
                    period2=period2[1:]
                if period1[0]=='T':
                    period1=period1[1:]   
                quant1=period1[:-1]
                quant2=period2[:-1]
                if mod1=='NA' and mod2=='NA':
                    if quant1==quant2:
                        return 1
                    else:
                        return 0
                else:
                    if abs(float(quant1)-float(quant2))<3:
                        return 1
                    else:
                        return 0
        else: # comparison in different units - convert everything to hours
            if mod1=='NA' and mod2=='NA':
                quant1=convert2hrs(period1)
                quant2=convert2hrs(period2)
                if quant1==quant2:
                    return 1
            else:
                if period1[0]=='T':
                    quant1=float(period1[1:-1])
                    prefix='T'
                    
                else:
                    quant1=float(period1[:-1])
                    prefix=''
                if quant1<2:
                    lower_bound=0.0
                else:
                    lower_bound=quant1-2     
                upper_bound=quant1+2
                quant2=convert2hrs(period2)
                if quant2<=convert2hrs(prefix+str(upper_bound)+unit1) and quant2> convert2hrs(prefix+str(lower_bound)+unit1):
                    return 1
                else:
                    return 0   
        return 0
    
    def convert2hrs(period):
        unit = period[-1]
        period=period[:-1]
        if unit == 'M':
            if period[0]=='T': #minutes
                quant=float(period[1:])/60
            else: #month
                quant=float(period)*4*7*24
        else:
            if period[0]=='T': 
                period=period[1:]
            if unit == 'D':
                quant=float(period)*24
            elif unit == 'W':
                quant=float(period)*24*7
            elif unit == 'Y':
                quant=float(period)*24*4*7*12
            elif unit == 'H':
                quant=float(period)
            elif unit == 'S':
                quant=float(period)/2400
        return quant
            
    
    def compare_timex(text_fname1, text_fname2, option, dic):     
        '''
        Check whether the TIMEX3s in text_fname1 can be found in text_fname2:
        
        args:
            text_fname1: filename of the first xml file 
            text_fname2: filename of the first xml file
            option:      exact, overlap or partialCredit match
            dic:         a dictionary that maps events id in the first xml file
                     to the corresponding id in second one
        
        Output:
            totalTimex: total number of TIMEX3 in the first file
            matchTimex: total number of TIMEX3 in the first file that can be found in the second file
            tspanPartcialCredit: same as above, but discount overlap TIMEX3 matches (as 0.5), and exact match as 1
            ttyp:         number of correct type in the first file
            tval:         number of correct val in the first file
            tmod:         number of correct mod in the first file
            dic:         a dictionary that maps events/timex id in the first xml file
                         to the corresponding id in second one
        '''
        timexes1=get_timex(text_fname1)
        timexes2=get_timex(text_fname2)
        totalTimex=len(timexes1)
        tspanPartcialCredit=0
        matched_ids={}
        #timex_attr+scores: {id: [val, mod, type, exact_match, id2]}
        timex_attr_scores={}
        matchTimex=0
        ttyp=0
        tval=0
        tmod=0
        for timex1 in timexes1:            
            id1, startStr1, endStr1, text1, type1, val1, mod1=timex1
            id2, startStr2, endStr2, text2, type2, val2, mod2=["", "", "", "", "", "", ""]
            start1=int(startStr1)
            end1=int(endStr1)
            dic[id1]=''
            timex_attr_scores[id1]=[0,0,0,0,'']
            for timex2 in timexes2:
                spanScore=0
                compare_flag=0
                val=0
                mod=0
                attr_type=0
                if timex2<>['']:
                    start2=int(timex2[1])
                    end2=int(timex2[2])
                    id2, startStr2, endStr2, text2, type2, val2, mod2=timex2
                    words1=text1.split()
                    words2=text2.split()
                    for punctuation in punctuations:
                        while punctuation in words1:
                            words1.remove(punctuation)
                        while punctuation in words2:
                            words2.remove(punctuation)
                    if (not re.search('\w', text2)) or (not re.search('\w', text1)):
                        #if text1 or text2 only contains white spaces, it is considered as mismatch
                        spanScore=0
                    else:
                        if start1<=start2: 
                            if end1>=start2+1:
                                spanScore=0.5
                                if words1==words2:
                                    spanScore=1   
                        else:
                            if end2>start1+1:
                                spanScore=0.5
                                if words1==words2:
                                    spanScore=1                     
                        if spanScore>0:
                            if dic[id1]=='':
                                dic[id1]=id2
                            else:
                                dic[id1]=dic[id1]+'#@#'+id2
                            try:
                                matched=matched_ids[id2]
                            except KeyError:
                                matched='' 
                            if type1==type2:
                                attr_type=1
                            if type1.upper() in ['DURATION','FREQUENCY'] and type2.upper() in ['DURATION','FREQUENCY'] and val1<>'' and val2<>'':
                                duration=re.compile('^PT*\d*\.?\d*[YMWDHS]?$')
                                frequency=re.compile('^R\d*\.?\d*P?T?\d*\.?\d*[YMWDHS]?$')
                                if type1.upper() == 'FREQUENCY' and type2.upper()=='FREQUENCY':                                    
                                    if frequency.match(val1.upper()) and frequency.match(val2.upper()):
                                        val=DurationFrequencyValCompare(val1.upper(),mod1.rstrip(),val2.upper(),mod2.rstrip())
                                    else:
                                        val=0
                                if type1.upper() == 'DURATION' and type2.upper()=='DURATION':
                                    if duration.match(val2.upper()) and duration.match(val1.upper()):
                                        val=DurationFrequencyValCompare(val1.upper(),mod1.rstrip(),val2.upper(),mod2.rstrip())
                                    else:
                                        val=0                                        
                            else:
                                if val1.upper()==val2.upper():
                                    val=1
                                else:
                                    val=0                                       
                            if mod1==mod2:
                                mod=1  
                            if matched=='':
                                if val>timex_attr_scores[id1][0] or timex_attr_scores[id1][0]==0:
                                    previous_scores=timex_attr_scores[id1]
                                    timex_attr_scores[id1]=[val, mod, attr_type, spanScore, id2]
                                    matched_ids[id2]=id1
                                    if previous_scores[4]<>'':
                                        matched_ids[previous_scores[4]]='' 
                            else:
                                if timex_attr_scores[matched][3]<1:
                                    timex_attr_scores[matched]=[0,0,0,0,'']
                                    timex_attr_scores[id1]=[val, mod, attr_type, spanScore, id2]
                                    matched_ids[id2]=id1                       
        for timex1 in timexes1: 
            if timex1<>['']:                
                id1, startStr1, endStr1, text1, type1, val1, mod1=timex1
                tval+=timex_attr_scores[id1][0]
                tmod+=timex_attr_scores[id1][1]
                ttyp+=timex_attr_scores[id1][2]  
                tspanPartcialCredit+=timex_attr_scores[id1][3] 
                if dic[id1]=='':
                    dic[id1]=id1
                if option=="exact":
                    if timex_attr_scores[id1][3] > 0.5:
                        matchTimex+=1
                else:
                    if timex_attr_scores[id1][3] > 0:
                        matchTimex+=1                             
        return totalTimex, matchTimex, tspanPartcialCredit, ttyp, tval, tmod, dic
    
    def timexEvaluation(gold_fname, system_fname, option,goldDic={},systemDic={}):
        '''
        evaluate a system output xml file against its corresponding goldstandard file:
        
        Args:
            gold_fname:      filename of the gold standard xml file
            system_fname:    filename of the system output xml file
            option:          exact, overlap or partialCredit match
            goldDic:         a dictionary that map Event id in goldstardard to those
                             in the system output
            systemDic:       a dictionary that map Event id in system outpt to those
                             in the gold standard
        
        Output:
            goldDic:        a dictionary that map Event/Timex id in goldstardard to those
                            in the system output
            systemDic:      a dictionary that map Event/Timex id in system outpt to those
                            in the gold standard
            goldTimexCount: total number of Timex annotated in the gold standard
            systemTimexCount: total number of Timex marked in the system output
            precCount:      system matched Timex found in gold standard 
            recallCount:    gold standard matched Timex found in system 
            recalltype:     correct type count in gold standard matched Timex found in system 
            recallVal:      correct val count in gold standard matched Timex found in system 
            recallMod:      correct modifier count in gold standard matched Timex found in system 
            recallPC:       partial credit recall match
            precPC:         partial credit precision match
        '''
        #compute recall
        recallScores=compare_timex(gold_fname, system_fname,option, goldDic)
        goldTimexCount, recallCount, recallPC, recalltype, recallVal, recallMod, goldDic=recallScores
        
        #compute precision
        precisionScores=compare_timex(system_fname, gold_fname,option, systemDic)
        systemTimexCount, precCount, precPC, prectype, precVal, precMod, systemDi=precisionScores
        
        if goldTimexCount<>0:
            if option=='partialCredit':
                recall=float(recallPC)/goldTimexCount
            else:
                recall=float(recallCount)/goldTimexCount
        else:
            recall=0
        if systemTimexCount<>0:
            if option=='partialCredit':
                precision=float(precPC)/systemTimexCount
            else:
                precision=float(precCount)/systemTimexCount
        else:
            precision=0
        #attribute score: percentage of the correct attribute in total matched # of timex. same as temp eval 2
        if recallCount<>0:
            typeScore=float(recalltype)/recallCount
            valScore=float(recallVal)/recallCount
            modScore=float(recallMod)/recallCount
        else:
            typeScore=0.0
            valScore=0.0
            modScore=0.0
        if (goldTimexCount+systemTimexCount)>0:
            averagePR=((recall*goldTimexCount)+(precision*systemTimexCount))/(goldTimexCount+systemTimexCount)
        else:
            averagePR=0.0
        if (precision+recall)>0:
            fScore=2*(precision*recall)/(precision+recall) 
        else:
            fScore=0.0
        
        print("""
            Total number of TIMEX3s: 
               Gold Standard :\t\t"""+str(goldTimexCount)+"""
               System Output :\t\t"""+str(systemTimexCount)+"""
            --------------
            Precision :\t\t\t"""+'%.4f'%(precision)+"""
            Recall :\t\t\t""" + '%.4f'%(recall)+"""
            Average P&R : \t\t"""+'%.4f'%(averagePR)+"""
            F measure : \t\t"""+'%.4f'%(fScore)+"""
            --------------
            type match score :\t\t"""+'%.4f'%(typeScore)+"""
            Val match score :\t\t"""+'%.4f'%(valScore)+"""
            Mod match score :\t\t"""+'%.4f'%(modScore))
        
        goldDic['Admission']='Admission'
        goldDic['Discharge']='Discharge'
        systemDic['Admission']='Admission'
        systemDic['Discharge']='Discharge'    
        return goldDic, systemDic,goldTimexCount,systemTimexCount,precCount,recallCount,recalltype,recallVal,recallMod,recallPC,precPC
            
    
    if __name__ == '__main__':
        usage= "%prog [options] [goldstandard-file] [systemOutput-file]" + __doc__
        parser = argparse.ArgumentParser(description='Evaluate system output Timex3s against gold standard Timex3s.')
        parser.add_argument('gold_file', type=str, nargs=1, \
                            help='gold standard xml file')
        parser.add_argument('system_file', type=str, nargs=1,
                         help='system output xml file')
           
        args = parser.parse_args()
        
        # run on a single file
        if os.path.isfile(args.gold_file[0]) and os.path.isfile(args.system_file[0]):
            gold=args.gold_file[0]
            system=args.system_file[0]
            timexEvaluation(gold, system,'overlap')
            print "Warning: This script calculates overlapping timex span match between two files only. Please use the i2b2Evaluation.py script instead for more options."
        else:
            print "Error: Please use i2b2Evaluation.py for evaluating two directories"