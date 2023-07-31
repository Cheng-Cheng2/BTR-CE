import sys
#from tidylib import tidy_document
#sys.path.append('/data/chengc7/MarkerTRel')
#sys.path.append('/data/chengc7/MarkerTRel')

import xml.etree.ElementTree as ET
import tempfile
import os
import re




class evaluation:
    
	def __init__(self, doc_id, link_pred_dict, link_prob_dict, gold_xml_dir, predicted_xml_dir): #label_dict
		self.document_id = doc_id
		self.link_pred_dict = link_pred_dict
		self.link_prob_dict = link_prob_dict
		self.gold_xml_dir = gold_xml_dir
		self.predicted_xml_dir = predicted_xml_dir
	# def parseExample(self):
	# 	for ex in self.exs:
	# 		self.predicted_links[ex['lid']] = 
	def eval(self):
    
		gold_xml_file = open(os.path.join(self.gold_xml_dir, str(self.document_id)+'.xml'), 'r')
		lines = gold_xml_file.readlines()
		predicted_xml_file = open(os.path.join(self.predicted_xml_dir, str(self.document_id)+'.xml'), 'w')
		for line in lines:
			if "<TLINK" not in line: 
			#elif "</TAGS>" not in line:# and "<TLINK" not in line:
				predicted_xml_file.write(line)
			else:
    			#l = <TLINK id="SECTIME35" fromID="E28" fromText="fevers" toID="Discharge" toText="Discharge" type="BEFORE" />
				m = re.search(r"(.*<TLINK id=\")(\w+)(\".*type=\")(\w+)(\".*)", line)
				tl_id = m.group(2) #self.link_pred_dict[tl_id]
				add_prob = f"\"prob={self.link_prob_dict[tl_id]}\""
				#newline = "".join(m.group(1,2,3)) + self.link_pred_dict[tl_id] + m.group(5) + '\n'
				newline = "".join(m.group(1,2,3)) + self.link_pred_dict[tl_id] + "\" " + add_prob + " />\n"
				predicted_xml_file.write(newline)
    
		predicted_xml_file.close()
	
     

	# def parseTags(self, tags):

	# 	for child in tags:
	# 		if child.tag == 'EVENT':
	# 			self.events[child.attrib['id']] = child.attrib['text']  

	# 		elif child.tag == 'TIMEX3':
	# 			self.timex3[child.attrib['id']] = child.attrib['text'] 

# cited from: https://github.com/yuyanislearning/CTRL-PG
# class psl_evaluation:

# 	def __init__(self, ex, xml_folder, dev=True): #label_dict

# 		# self.events = {}
# 		# self.timex3 = {}
# 		self.document_id = ex[ex]
# 		#self.label_dict = label_dict
# 		self.dev = dev
# 		self.xml_folder = xml_folder

# 	def eval(self, labels, event_ids, sen_ids, preds):

# 		xml_file = os.path.join(self.xml_folder, str(self.document_id)+'.xml')
# 		text=open(xml_file).read()
# 		text=re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+",u"",text)
# 		text=re.sub('&', ' ', text)
# 		#print(text)
# 		root = ET.fromstring(text)
# 		self.parseTags(root[1])

	
# 		gold_xml_file = open(os.path.join(self.xml_folder, str(self.document_id)+'.xml'), 'r')
# 		lines = gold_xml_file.readlines()
# 		predicted_xml_file = open(os.path.join(self.xml_folder, str(self.document_id)+'.xml'), 'w')
# 		for line in lines:
# 			if "<TLINK" not in line: 
# 			#elif "</TAGS>" not in line:# and "<TLINK" not in line:
# 				predicted_xml_file.write(line)
# 			else:
# 				#for i,([id1, id2], label, sen_id, pred) in enumerate(zip(event_ids, labels, sen_ids, preds)):

# 					#label = label_dict[label_ids]
# 					# event1 = (self.events[id1] if "E" in id1 else self.timex3[id1]).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
# 					# event2 = (self.events[id2] if "E" in id2 else self.timex3[id2]).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
				
#     			#l = <TLINK id="SECTIME35" fromID="E28" fromText="fevers" toID="Discharge" toText="Discharge" type="BEFORE" />
# 				m = re.search(r"(<TLINK id=)(\"\w+\")(.*type=)(\"\w+\")", line)
# 				tl_id = m.group(2)
# 				newline = m.group(1) + m.group(2) + m.group(3) + ex['tl_id']['predicted_label'] + '\n'
# 				predicted_xml_file.write(newline)
# 				#predicted_xml_file.write('<TLINK id="TL{}" fromID="{}" fromText="{}" toID="{}" toText="{}" type="{}" senid="{}" pred="{}" />'.format(str(i+1), id1, event1, id2, event2, label.upper(), sen_id[1], pred) + '\n')
# 				#predicted_xml_file.write(line)
# 		predicted_xml_file.close()
		

	# def parseTags(self, tags):

		# for child in tags:
		# 	if child.tag == 'EVENT':
		# 		self.events[child.attrib['id']] = child.attrib['text']  

		# 	elif child.tag == 'TIMEX3':
		# 		self.timex3[child.attrib['id']] = child.attrib['text'] 

		#print(self.events)
		#print(self.timex3)
