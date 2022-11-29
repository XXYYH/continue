import os
from main_cp import *

d = pickle.load(open(path_dictionary, 'rb'))
event2word, word2event = d
for key in word2event:
    print(key)
print("dic: ", d)