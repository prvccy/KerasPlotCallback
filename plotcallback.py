# Copyright 2018 prvccy <prvccy@gmail.com>
# All rights reserved. MIT License.


from IPython.display import clear_output
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
sns.set(style='whitegrid', palette="muted", color_codes=True) 
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix
import math
from keras.utils.np_utils import to_categorical


class ClassificationPlotCallback(Callback):
    def __init__(self, evas=['auc','f1','confusion_matrix']):
        self.cell_size = (6, 4)
        self.max_cols = 2
        self.percentile = percentile
        self.metrics = []
        self.aucs = [] if 'auc' in evas else None
        self.f1_score = [] if 'f1' in evas else None
        self.recall = [] if 'f1' in evas else None
        self.precision = [] if 'f1' in evas else None
        self.confusion_matrix = 'confusion_matrix' in evas


    def on_epoch_end(self, epoch, logs={}):
        self.metrics.append(logs.copy())
        clear_output(wait=True)
        plots = []
        for k in self.metrics[0]:
            if k.startswith('val_'):
                continue
            plots.append(k)

        y_pred = self.model.predict(self.validation_data[0])
        if self.y_pred.shape[1]>1:
            y_pred = to_categorical(self.y_pred.argmax(axis=1))
        else:
            y_pred = self.y_pred.round(0)
        
        if self.aucs is not None:
            self.aucs.append(roc_auc_score(self.validation_data[1], y_pred))
            plots.append('auc')
        
        if self.f1_score is not None:
            self.f1_score.append(f1_score(self.validation_data[1], y_pred))
            self.recall.append(recall_score(self.validation_data[1], y_pred))
            self.precision.append(precision_score(self.validation_data[1], y_pred))
            plots.append('f1-score')
            
        if self.confusion_matrix:
            if y_pred.shape[1]>1:
                cm = confusion_matrix(self.validation_data[1].argmax(axis=1), y_pred.argmax(axis=1))
            else:
                cm = confusion_matrix(self.validation_data[1], y_pred.round(0))
            plots.append('confusion_matrix')
            
        cols = self.max_cols
        if 'confusion_matrix' in plots and y_pred.shape[1] < 9:
            rows = math.ceil( len(plots) / self.max_cols)
        else:
            plotd = plots.copy()
            plotd.remove('confusion_matrix')
            rows = math.ceil( len(plotd) / self.max_cols) + 2
            
        fig = plt.figure(figsize=(self.cell_size[0] * cols, self.cell_size[1] * rows))
        gs = GridSpec(rows, cols)
        
        subplot_No = 0
        for p in plots:
            row = math.floor(subplot_No/self.max_cols)
            col = subplot_No - row * self.max_cols
            subplot_No += 1
            
            if p == 'confusion_matrix' and y_pred.shape[1] > 8:
                plt.subplot(gs[-2:, :])
            else:
                plt.subplot(gs[row, col])
            
            
            if p == 'confusion_matrix':
                sns.heatmap(pd.DataFrame(cm, range(len(cm)), range(len(cm)) ), annot=True, cmap="YlGnBu") 
                continue
            
            plt.xticks(range(1, self.params['epochs'] + 1))
            plt.xlim(1, self.params['epochs'])
            plt.locator_params(axis='x', nbins=10)
            
            # plot loss and metrics in model
            if p in self.metrics[0]:
                plt.plot(range(1, len(self.metrics) + 1), [e[p] for e in self.metrics], label=p)
                if 'val_'+p in self.metrics[0]:
                    plt.plot(range(1, len(self.metrics) + 1), [e['val_'+p] for e in self.metrics], label='val_'+p)
                    
            if p == 'auc':
                plt.plot(range(1, len(self.aucs) + 1), self.aucs, label="auc")
                
            if p == 'f1-score':
                plt.plot(range(1, len(self.f1_score) + 1), self.f1_score, label="f1-score")
                plt.plot(range(1, len(self.recall) + 1), self.recall, label="recall")
                plt.plot(range(1, len(self.precision) + 1), self.precision, label="precision")
                
            plt.legend(loc='center right')
        plt.tight_layout()
        plt.show();
