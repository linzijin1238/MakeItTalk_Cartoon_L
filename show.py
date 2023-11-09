import pickle

import matplotlib.pyplot as plt



pkl = open('E:/GitKraken/MakeItTalk_Cartoon/train\dump/autovc_align_train_fl.pickle', 'rb')

im = pickle.load(pkl)



plt.imshow(im)