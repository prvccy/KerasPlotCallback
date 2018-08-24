# Copyright 2018 prvccy <prvccy@gmail.com>
# All rights reserved. MIT License.


from keras.callbacks import Callback
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,precision_score,f1_score,roc_auc_score
import seaborn as sns
sns.set(style='whitegrid', palette="muted", color_codes=True) 

class PrvKCbkDataGen(object):
    '''
    generate data on epoch end
    self.data: dict
    level1 key: class, each class corresponding to one subplot
    level2 key: attribute
    level2 key: "plotclass" data will be plot using "plotclass"
    '''
    def __init__(self):
        self.data = dict()
        self.metrics = []
    
    def on_epoch_end(self, epoch, logs, predicts):
        raise NotImplementedError("on_epoch_end")
    
    def getData(self):
        return self.data

class PrvChart(object):
    '''
    plot data using matplotlib
    '''
    def plot(self):
        raise NotImplementedError("on_epoch_end")


class PrvKCbkController(object):
    '''
    use generated data to do sth useful, like earlystop savebestmodel
    '''
    def on_epoch_end(self, data={}):
        raise NotImplementedError("on_epoch_end")
        
    def on_training_end(self, data={}):
        pass

class PrvKerasCbk(Callback):
    '''
    callback add to callbacks list of keras.model.fit()
    
    this class will plot loss and metrics in keras.model.compile() as default
        e.g.  keras.model.compile(loss = 'mse', metrics = 'mae')
              keras.model.fit(callbacks = PrvKerasCbk() )
    
    plot confusion matrix: (loss must be binary_crossentropy or categorical_crossentropy)
        keras.model.fit(callbacks = PrvKerasCbk(datagens=[PCDGConfusionMatrix()]) )
        
    save best model and training curve：
        keras.model.fit(callbacks = PrvKerasCbk(controllers=[PKCSaveModelAndResult(path='/path/')]) )
    
    '''
    def __init__(self, datagens=[], controllers=[], plotcols = 2, subplotsize = (6,4)):
        self.datagens = datagens
        loss = PCDGLossAndMetrics()
        self.datagens.insert(0,loss)
        self.controllers = controllers
        self.plotcols = plotcols
        self.subplotsize = subplotsize
        self.data = dict()

    def on_epoch_end(self, epoch, logs={}):
        clear_output(wait=True)
        predicts = dict()
        # TODO handle no testset error
        predicts['test_y'] = self.validation_data[1]
        predicts['test_pred'] = self.model.predict(self.validation_data[0])
        
        # first run data generator
        for dg in self.datagens:
            dg.on_epoch_end(epoch, logs, predicts)
            self.data.update(dg.getData())

        # then plot
        numofsubplot = 0
        for p in self.data:
            if 'plotclass' in self.data[p]:
                numofsubplot += 1

        self.rows = numofsubplot//self.plotcols + int(numofsubplot%self.plotcols>0)
        fig, axes = plt.subplots(nrows=self.rows, ncols=self.plotcols, figsize=(self.subplotsize[0] * self.plotcols, self.subplotsize[1] * self.rows)) #, sharex=True 
        axs = axes.flatten()
        
        plt.xticks(range(1, self.params['epochs'] + 1))
        plt.xlim(1, self.params['epochs'])
        plt.locator_params(axis='x', nbins=10)
        
        subplot_No = 0
        for p in self.data:
            if 'plotclass' in self.data[p]:
                subplot = eval(self.data[p]['plotclass']+'()')
                subplot.plot(axs[subplot_No], self.data[p], p)
                subplot_No += 1
                
        plt.tight_layout()
        plt.show();
        # last run controller
        for c in self.controllers:
            c.on_epoch_end(self.data)
            if epoch == self.params['epochs']:
                c.on_training_end(self.data)


class PrvLineChart(PrvChart):
    ''' draw line chart '''
    def plot(self, ax, plotdata, p):
        for key in plotdata:
            if key != 'plotclass':
                ax.plot(range(1, len(plotdata[key])+1), plotdata[key], label=key)
                ax.set_title(p,fontsize=12,color='r')
        ax.legend()


class PCDGLossAndMetrics(PrvKCbkDataGen):
    def on_epoch_end(self, epoch, logs, predicts):
        self.metrics.append(logs.copy())
        for k in self.metrics[0]:
            if not k.startswith('val_'):
                if k not in self.data:
                    self.data.update({k:{}})
                self.data[k][k] = [epoch[k] for epoch in self.metrics]
                self.data[k]['val_'+k] = [epoch['val_'+k] for epoch in self.metrics]
                self.data[k]['plotclass'] = 'PrvLineChart'


class PCDGConfusionMatrix(PrvKCbkDataGen):
    def on_epoch_end(self, epoch, logs, predicts):
        for dataset in ['test']:
            if not dataset+'_y' in predicts:
                continue
            self.data.update({'ConfusionMatrix_' + dataset:{}})
            if predicts[dataset + '_pred'].shape[1]>1:
                cm = confusion_matrix(predicts[dataset + '_y'].argmax(axis=1), predicts[dataset + '_pred'].argmax(axis=1))
            else:
                cm = confusion_matrix(predicts[dataset + '_y'], predicts[dataset + '_pred'].round(0))
            self.data['ConfusionMatrix_' + dataset]['ConfusionMatrix_' + dataset] = cm
            self.data['ConfusionMatrix_' + dataset]['plotclass'] = 'PrvHeatMap'


class PrvHeatMap(PrvChart):
    def plot(self, ax,  plotdata, p):
        for key in plotdata:
            if key != 'plotclass':
                cm = plotdata[key]
                sns.heatmap(pd.DataFrame(cm, range(len(cm)), range(len(cm)) ), annot=True, cmap="YlGnBu", ax = ax) 


class PCDGF1Score(PrvKCbkDataGen):
    f1_score = []
    recall = []
    precision = []
    def on_epoch_end(self, epoch, logs, predicts):
        for dataset in ['test']:
            if not dataset+'_y' in predicts:
                continue
            self.f1_score.append(f1_score(predicts[dataset + '_y'], predicts[dataset + '_pred'],average='samples'))
            self.recall.append(recall_score(predicts[dataset + '_y'], predicts[dataset + '_pred'],average='samples'))
            self.precision.append(precision_score(predicts[dataset + '_y'], predicts[dataset + '_pred'],average='samples'))

            self.data.update({'F1':{}})
            self.data['F1']['F1'] = self.f1_score
            self.data['F1']['recall'] = self.recall
            self.data['F1']['precision'] = self.precision
            self.data['F1']['plotclass'] = 'PrvLineChart'
             
                
class PCDGROCAUC(PrvKCbkDataGen):
    aucs = []
    def on_epoch_end(self, epoch, logs, predicts):
        for dataset in ['test']:
            if not dataset+'_y' in predicts:
                continue
            self.aucs.append(roc_auc_score(predicts[dataset + '_y'], predicts[dataset + '_pred']))

            self.data.update({'AUC':{}})
            self.data['AUC']['AUC'] = self.aucs
            self.data['AUC']['plotclass'] = 'PrvLineChart'


class PKCSaveModelAndResult(PrvKCbkController):
    def __init__(self, path):
        now = datetime.datetime.now()
        self.savefile = path + '/' + now.strftime('%Y%m%d-%H:%M:%S:')+str(now.microsecond)
        
    def on_epoch_end(self, epoch, data={}):
        #save best model
        if data['loss']['val_loss'][-1] == min(data['loss']['val_loss']):
            files = os.listdir(self.path)
            for file in files:
                if file.startswith(self.savefile) and file.endswith('model'):
                    os.remove(self.path+file)
            self.model.save(self.savefile + '_loss' + str(data['loss']['val_loss'][-1]) + '.model')
        
    def on_training_end(self, data={}):
        #save training data
        with h5py.File(self.path+self.savefile, 'w') as hf:
            for key1 in self.data.keys():
                for key2 in self.data[key1].keys():
                    try:
                        if self.data[key1][key2] == None:
                            hf[key1+'/'+key2] = 0
                        else:
                            hf[key1+'/'+key2] = self.data[key1][key2]
                    except TypeError:
                        self.data[key1][key2].to_hdf(self.path+self.savefile,key1+'/'+key2)
            

                
