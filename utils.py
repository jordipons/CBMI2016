import theano, lasagne, csv
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import buildArchitecture as buildArch

def visualizeWcnn1(name):
    'Visualize weights of the convolutional layer of cnn1'

    ##!!## biases not shown !
    ##!!## deterministic W ?

    # # load parameters
    with open(name+'.param', 'rb') as paramFile:
        params = csv.reader(paramFile, delimiter='-')
        count=0;
        for param in params:
            if count==0:
                tmp1=param
                count=count+1
            else:
                tmp2=param
    parametersDL = {}
    parametersDL['type']=tmp2[tmp1.index('type')]
    parametersDS = {}
    parametersDS['numChannels']=tmp2[tmp1.index('numChannels')]
    parametersDS['yInput']=tmp2[tmp1.index('yInput')]
    parametersDS['xInput']=tmp2[tmp1.index('xInput')]
    parametersDS['numOutputNeurons']=tmp2[tmp1.index('numOutputNeurons')]
    parameters={}
    parameters['task']=tmp2[tmp1.index('task')]
    parameters['DS']=parametersDS
    parameters['DL']=parametersDL

    print("Building network..")
    input_var = T.tensor4('inputs')
    network,netLayers,parameters=buildArch.buildNet(input_var,parameters)
    # load trained network
    with np.load(name+'.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    print("Compiling functions..")
    # visualize convLayers
    conv_w = theano.function([],netLayers['2'].W)
    weights=conv_w()

    # plot W!
    for i in range(len(weights)):
        #plt.subplot(1,len(weights), i+1)
        plt.subplot(len(weights), 1, i+1)
        #plt.imshow(np.squeeze(weights[i]), cmap='seismic', interpolation='None', aspect='auto')
        plt.plot(np.squeeze(weights[i]))
        import essentia
        import essentia.streaming as es
        alloro=essentia.array(np.squeeze(weights[i]))
        input=es.VectorInput(alloro)
        bpmhist=es.BpmHistogram(maxBpm=220,minBpm=60)
        pool = essentia.Pool()
        input.data >> bpmhist.novelty
        bpmhist.bpm >> (pool, 'bpm')
        bpmhist.bpmCandidates >> (pool, 'bpmCandidates')
        bpmhist.bpmMagnitudes >> (pool, 'bpmMagnitudes')
        bpmhist.tempogram >> (pool, 'tempogram')
        bpmhist.frameBpms >> (pool, 'frameBpms')
        bpmhist.ticks >> (pool, 'ticks')
        bpmhist.ticksMagnitude >> (pool, 'ticksMagnitude')
        bpmhist.sinusoid >> (pool, 'sinusoid')

        essentia.run(input)
        print pool['bpm'],pool['bpmCandidates'],pool['bpmMagnitudes']#,pool['tempogram'],pool['frameBpms']

        plt.yticks(np.arange(-1, 1, 1))
    #plt.colorbar()
    plt.show()

def trainingEvolution(name):
    'Plot the training evolution: training loss, validation loss and validation accuracy.'

    # load data
    df = pd.read_csv(name+'.training')
    trainingLoss = df['trainingLoss']
    validationLoss = df['validationLoss']
    validationAccuracy = df['validationAccuracy']

    # plot training evolution!
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(range(1,len(trainingLoss)+1,1),trainingLoss, color='red',linestyle='--', marker='o',label="Training Loss")
    plt.hold(True)
    plt.plot(range(1,len(trainingLoss)+1,1),validationLoss,color='blue',linestyle='--', marker='o',label="Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Validation Accuracy (%)')
    plt.plot(range(1,len(trainingLoss)+1,1),validationAccuracy,color='blue',linestyle='--', marker='o')

    plt.show()

def visualizeActivations(a,title):
    import matplotlib.pyplot as plt;
    for i in xrange(a.shape[1]):
        plt.subplot(2,np.ceil((a.shape[1])/2),i+1)
        plt.imshow(a[0][i], interpolation='None')#, aspect='auto');
        plt.title('Av: '+str("{0:.4f}".format(np.mean(np.mean(np.mean(a,axis=0),axis=2),axis=1)[i]))+' Mx: '+str("{0:.2f}".format(np.max(np.max(np.max(a,axis=0),axis=2),axis=1)[i])))
        plt.colorbar();
    plt.suptitle(title)
    plt.show()

name='/home/jpons/Dropbox/PhD-MTG/DeepLearning/28-01-2016/MIRdeepLearning/data/results/Ballroom_cnn1_138918266775421553791267201726964333588'
visualizeWcnn1(name)
#trainingEvolution(name)

##!## missing utils:
# - gradient propagation