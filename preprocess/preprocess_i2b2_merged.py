# -*- coding: utf-8 -*-

'''
citation: adapted from code om https://github.com/chentao1999/MedicalRelationExtraction/blob/master/data_process_i2b2.py                                                                           \
'''

import xml.etree.ElementTree as ET
import random
import os, pdb, glob, json

from pathlib import Path
from attr import attrs
import traceback


from dataclasses import dataclass
from typing import (List, Iterator, Tuple, Union, Dict, Mapping, Callable, Set, Type, TextIO, Optional)
from collections import deque, OrderedDict

#TRAINNING_DATA_DIR = "./corpus/i2b2/2012-07-15.original-annotation.release/"
TRAIN_DATA_DIR = "./corpus/i2b2/train_merged"
TEST_DATA_DIR = "./corpus/i2b2/ground_truth/merged_xml/"
SAVE_DIR = "./corpus/i2b2/"



@dataclass
class I2B2Entity():
    id: str
    eType: str
    text: str
    type: str
    val: str
    span: Tuple[int, int]

    @classmethod
    def new(self, cls):
        #breakpoint()
        res = {}
        res['span'] = [int(cls.attrib['start']) - 1, int(cls.attrib['end']) - 1] # dif from tbdense
        res['id'] = cls.attrib['id']
        res['eType'] = cls.tag #cls.attrib['type'] #cls.tag
        res['text'] = cls.attrib['text']
        res['type'] = cls.attrib['type']
        if cls.tag == "EVENT":
            # DATA: <EVENT id="E80" start="895" end="908" text="normal saline" modality="FACTUAL" polarity="POS" type="TREATMENT" />

            res['val'] = None  
        elif cls.tag == "TIMEX3":
            
            # DATA: <TIMEX3 id="T13" start="3174" end="3181" text="2/29/00" type="DATE" val="2000-02-29" mod="NA" />

            res['val'] = cls.attrib['val'] # different in TIMEX'
        else: # "SECTIME"
            # DATA: <SECTIME id="S0" start="18" end="28" text="07/03/1999" type="ADMISSION" dvalue="1999-07-03" />

            res['val'] = cls.attrib['dvalue'] # different in SECTIME'
        return I2B2Entity(**res)
    
@dataclass
class I2B2Relation():
    id: str
    tlType: str
    fromID: str
    toID: str
    fromText: str
    toText:str
    type: str
    secType: str
    #properties: Dict[str, List[I2B2Entity]] #Dict[str, List[Union[str, I2B2Entity]]]
    
    def new(cls, entities):
        
        res = {}
        # relation type
        
        #res['properties'] = {}
        res['id'] = cls.attrib['id']
        #res['type'] = cls.attrib['type']
        res['fromID'] = cls.attrib['fromID']
        res['toID'] = cls.attrib['toID']
        res['fromText'] = cls.attrib['fromText']
        res['toText'] = cls.attrib['toText']
        res['type'] = cls.attrib['type']
        res['secType'] = ""
        #try: 
            #res['properties']['type'] = cls.attrib['type']
            # res['properties']['source'] = entities[res['fromID']]
            # res['properties']['target'] = entities[res['toID']]
        # e1_id, e2_id = res[]
        # except KeyError as e :
        admit_time, discharge_time = "", ""
        if 'sectime' in res['id'].lower():
            if res['toID'] in ['Discharge', 'Admission']: # corrected below
                res['secType'] = "ADMISSION" if res['toID']=='Admission' else "DISCHARGE"
            else:
                e1_type, e2_type = entities[res['fromID']].eType, entities[res['toID']].eType
                if 'S0' in entities:
                    admit_time = entities['S0'].text
                if 'S1' in entities:
                    discharge_time = entities['S1'].text
                # else:
                #     print("other case")
                    #pdb.set_trace()
                if e1_type == 'EVENT' and e2_type=='TIMEX3':
                    #print("ok")
                    t2 = entities[res['toID']].text
                    if t2==admit_time:
                        res['secType'] = "ADMISSION"
                    elif t2==discharge_time:
                        res['secType'] = "DISCHARGE"
                    else:
                        print("other case")
                        #pdb.set_trace()

                        #
                else:
                    print("debug other case")
                    
                    #pdb.set_trace()
                    #pass # this was tested to never happen

                        
        if res['toID'] in ['Discharge', 'Admission']: # 781
            for k, v in entities.items():
                #pdb.set_trace()
                if v.text==res['toID']: #in ['Discharge', 'Admission']:
                    res['toID'] = v.id
                    #res['properties']['target'] = entities[res['toID']]
                    break
                elif v.text in ['discharged']:
                    # this only hits for 173.xml
                    res['toID'] = v.id
                    #res['properties']['target'] = entities[res['toID']]
                    break
        # NOTE: for sectime this was tested so that only possibile outliers are ['Discharge', 'Admission']

                
        e1_type, e2_type = entities[res['fromID']].eType, entities[res['toID']].eType

        try:
            if e1_type == 'EVENT' == e2_type:
                res['tlType'] = 'ee'
            elif e1_type == 'TIMEX3' == e2_type:
                res['tlType'] = 'tt'
            elif (e1_type == 'TIMEX3') and (e2_type == 'EVENT'):
                res['tlType'] = 'te'
            elif (e1_type == 'EVENT') and (e2_type == 'TIMEX3'):
                res['tlType'] = 'et'
            elif (e1_type == 'SECTIME') and (e2_type == 'EVENT'):
                res['tlType'] = 'se'
            elif (e1_type == 'EVEHT') and (e2_type == 'SECTIME'):
                res['tlType'] = 'es'
            else:
                print("other form of sect tl link exists?")
                pdb.set_trace()

        except Exception as e:
            print(f"trackback: {traceback.format_exc()}")

            print(f"error msg: {repr(e)}")

            print("res:", res)
            pdb.set_trace()
        
        # debug:
        if e1_type == 'SECTIME' or e2_type =='SECTIME':
            pdb.set_trace() # this never happens, I get now
            """
            e.g. 367.xml 
            <SECTIME id="S0" start="18" end="28" text="2013-10-21" type="ADMISSION" dvalue="2013-10-21" />
            <SECTIME id="S1" start="46" end="56" text="2013-10-31" type="DISCHARGE" dvalue="2013-10-31" />
            
            but the sectime links does not point to this sectime events, they point to time events instead:
            <TLINK id="SECTIME2" fromID="E21" fromText="systemic emboli" toID="T2" toText="2013-10-31" type="BEFORE" />

            """
        if 'SECTIME' in res['id']:
            #pdb.set_trace()
            if e1_type == 'TIMEX3' and e2_type == 'EVENT':
                res['tlType'] = 'se'
            elif e2_type == 'TIMEX3' and e1_type == 'EVENT':
                res['tlType'] = 'es'
            else:
                if res['toText'] in ['Discharge', 'Admission']:
                    res['tlType'] = 'es'
                else:
                    pdb.set_trace()
            
        return I2B2Relation(**res)    


@dataclass
class I2B2Doc:
    id: str
    raw_text: str = ""
    entities: Mapping[str, I2B2Entity] = None
    relations: Mapping[str, I2B2Relation] = None

    def parse_entities(self, raw_text, event_fts={}):

        res = []
        
        # this new dataset doesn't have span indicators, so we need to create it manually
        #q, raw_text = self.build_queue(all_text)
        #breakpoint()
        for e in event_fts:
            
            #m = q.popleft()
            # while m[0] != e.text:
            #     m = q.popleft()
            #entity = TBDEntity.new(e, event_fts, m[1])
            entity = I2B2Entity.new(e)
            # make sure span created is correct
            #breakpoint()
            try:
                assert raw_text[entity.span[0] : entity.span[1]] == entity.text
            except AssertionError:
                print(f"raw_text:{raw_text[entity.span[0] : entity.span[1]]}")
                print(f"entity.text:{entity.text}")
                #pdb.set_trace()
                escapes = ["&apos;", "&quot;"]
                still_pb = False
                for esc in escapes:
                    still_pb = False if esc in raw_text else still_pb
                if still_pb:
                    pdb.set_trace()

            res.append(entity)
            
        return res, raw_text
    
    # def build_queue(self, all_text):
    #     #breakpoint()
    #     raw_text = ""
    #     counter = -1
    #     q = deque()
    #     for tx in all_text:
    #         q.append((tx, counter))
    #         counter += len(tx)
    #         raw_text += tx

    #     return q, raw_text

    def parse(self, id: str, text_path: str):
        
        print('file id', id)
        
        #pdb.set_trace()
        with open(text_path) as f:

            tree = ET.parse(f)
            root = tree.getroot()
            # REVIEW: is the removing \n causing spacy recognition errors?
            all_text = root.find("TEXT").text.replace("\n", " ").strip()
            #all_text = root.find("TEXT").text
            ###
            tags = root.find("TAGS")
            events_ft = tags.findall("EVENT")
            timexs_ft = tags.findall("TIMEX3")
            sects_ft = tags.findall("SECTIME")
            
            tlinks_ft = tags.findall("TLINK")
            
            #events = xml_tree.xpath('.//EVENT')
            #timex = xml_tree.xpath('.//TIMEX3')
            #timex = [t for t in xml_tree.xpath('.//TIMEX3') if t.attrib['functionInDocument'] != 'CREATION_TIME']
            
            #event_fts = {e.attrib['eventID']: (e.attrib['tense'], e.attrib['polarity'], e.attrib['eiid']) for e in xml_tree.xpath('.//MAKEINSTANCE')}
            #timex_fts
            
            events, raw_text = self.parse_entities(all_text, events_ft)
            timexs, _ = self.parse_entities(all_text, timexs_ft)
            sects, _ = self.parse_entities(all_text, sects_ft)

            
            
            entities = events + timexs + sects
            entities = OrderedDict([(e.id, e) for e in entities])

            relations = []
            #pos_pairs = []

            total_count = 0
            missing_count = 0
            
            timex = 0
            id_counter = 1
            
            #breakpoint()
            for tl in tlinks_ft:
                # # It seems t0 in the TBDense dataset denote doc_create_time
                # if (pair[0][0] == 't' or pair[1][0] == 't') or (pair[0] == 't0' or pair[1] == 't0'):
                #     #print('Doc Time, skip')
                #     #print(pair)
                #     timex += 1
                # else:
                relations.append(I2B2Relation.new(tl, entities))
                id_counter += 1


            #relations = {r.id: r for r in relations} # DEBUG: changed to below for duplicated link id
            duplicate = 0
            formulate_relations = {}
            for r in relations:
                if r.id in formulate_relations:
                    duplicate += 0
                    new_id = f"{r.id}_{duplicate}"
                    formulate_relations[new_id] = r
                else:
                    formulate_relations[r.id] = r
            return I2B2Doc(id, raw_text, entities, formulate_relations)
        

        
        
#############above are added by me
# def file_name(file_dir):
#     L=[]
#     for root, dirs, files in os.walk(file_dir):
#         for file in files:
#             if os.path.splitext(file)[1] == '.xml':
#                 L.append(os.path.join(root, file))
#                 # L.append(file)
#     return L


@attrs(auto_attribs=True, slots=True)
class JsonSerializer:
    types: Set[Type]

    def __call__(self, obj):
        if type(obj) is list:
            return [self(x) for x in obj]
        elif type(obj) is dict:
            return {k: self(v) for k, v in obj.items()}
        elif type(obj) in self.types:
            return self(vars(obj))
        return obj
    
    
    
# NOTE: out version to make it match PEER input
def data_process(inDIR, outFile):
    #fileList = file_name(inDIR)
    fileList = glob.glob(os.path.join(inDIR, "*"))
    # print(len(fileList))
    # lableType = set()
    outFile = open(outFile, "w")
    # # NOTE: for every single .xml file
    #pdb.set_trace()
    for i, f in enumerate(fileList):
                #for src_file, doc_id in src_id_map.items():
        print(f"{i}/{len(fileList)} total files")
        doc_id = f.split("/")[-1].split('.xml')[:-1]
        doc = I2B2Doc(doc_id)
        doc = doc.parse(doc_id, f)
        serializer = JsonSerializer({I2B2Entity, I2B2Relation, I2B2Doc})
        line = json.dumps(doc, ensure_ascii=False, default=serializer)
        
        outFile.write(line)
        outFile.write('\n')
        
        
        #break



def pos_process(json_file):
    # lines = []
    # for line in json.loads(open(json_file)):
    #     line = line.replace("&apos;", "'")
    #     line = line.replace("&quot;", "\"")
    #     lines.append(line)
    
    return
    
    
    
if __name__ == '__main__':
    data_process( TRAIN_DATA_DIR , SAVE_DIR + "train.json")
    data_process( TEST_DATA_DIR, SAVE_DIR + "test.json")


    # pos_process(SAVE_DIR + "train.json")
    # pos_process(SAVE_DIR + "test.json")
    
