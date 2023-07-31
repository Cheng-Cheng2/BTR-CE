# MARKERS_I2B2=["ner", "etype", "ner_plus_time"]
# MODELS=["bert-base-uncased"]
# CONTEXT_WINDOWS=[0]
# LRS=["1e-5", "2e-5", "1e-4"]
# SEEDS=[1]



#MARKERS_I2B2=["ner", "ner_plus_time", "ner", "pos", "dep", "ner_dep", "ner_pos", "ner_dep_pos", "ner_time_dep", "ner_time_dep_pos"]
MARKERS_I2B2=["ner", "ner_plus_time", "ner_time_pos", "ner_pos", "bert"]
MODELS=["bert-base-uncased"]
CONTEXT_WINDOWS=[0, 1, 2, 3]
LRS=["2e-5"]
SEEDS=[1, 2, 3, 4, 5]
TRAIN_BATCH_SIZE = [16]

MARKERS_TBD = ["pos", "pos_tense", "pos_tense_polarity", "bert"]
#MODELS=["bert-base-uncased", "roberta-base"]



