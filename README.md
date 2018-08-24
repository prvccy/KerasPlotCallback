# KerasPlotCallback
plot loss evaluation metrics F1-score AUC and confusion matrix on the end of epoch


usage:

    plot loss and metrics in keras.model.compile() as default
        e.g.  keras.model.compile(loss = 'mse', metrics = 'mae')
              keras.model.fit(callbacks = PrvKerasCbk() )
    
    add confusion matrix: (loss must be binary_crossentropy or categorical_crossentropy)
        keras.model.fit(callbacks = PrvKerasCbk(datagens=[PCDGConfusionMatrix()]) )
        
    save best model(min loss) and training curve(loss and metrics) in h5 file：
        keras.model.fit(callbacks = PrvKerasCbk(controllers=[PKCSaveModelAndResult(path='/path/')]) )
