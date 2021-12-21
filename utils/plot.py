from matplotlib import pyplot as plt

def getAccuracyPlot(history, model_name): #Plots accuracy
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(accuracy)
    plt.plot(val_accuracy)
    plt.title(model_name + ' Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['accuracy','val_accuracy'], loc='upper left')
    plt.show()
    
def getLossPlot(history, model_name): #Plots loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(model_name + ' Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['loss','val_loss'], loc='upper left')
    plt.show()


