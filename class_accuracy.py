import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
np.set_printoptions(linewidth=140, edgeitems = 101)
parser = argparse.ArgumentParser(description="class accuracy")
parser.add_argument('--score-file', type=str, default = "", help='score file.')

args = parser.parse_args()
ScoreFilePath = ""

if args.score_file != "":
	score = np.load(args.score_file)
else:
	score = np.load(ScoreFilePath)
print(score)
print(np.diag(score) / score.sum(axis=1))
print("Overall accuracy: %.2f" % (np.sum(np.diag(score)) / np.sum(score.sum(axis=1)) * 100))


df_cm = pd.DataFrame(score[70:, :].astype(int), range(70, 101), range(101))
fig, ax = plt.subplots(figsize=(20,10))
ax.xaxis.tick_top()
#sn.set(font_scale=0.2) # for label size
sn.heatmap(df_cm, xticklabels=1, yticklabels=1, cmap = plt.cm.Reds, cbar = False, annot=True, fmt = 'd', annot_kws={"size": 10}, ax = ax)
plt.savefig('./cm3', dpi = 500)
#plt.show()
