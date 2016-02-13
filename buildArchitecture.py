import lasagne

def buildNet(input_var, parameters):
    'Select a deep learning architecture'

    if parameters['DL']['type']=='blackbox':
        return blackbox(input_var, parameters)
    if parameters['DL']['type']=='time' or parameters['DL']['type']=='frequency':
        return timeFrequency(input_var, parameters)
    if parameters['DL']['type']=='mergeTimeFrequency':
        return mergeTimeFrequency(input_var, parameters)
    if parameters['DL']['type']=='loadMergeTimeFrequency':
        return loadMergeTimeFrequency(input_var, parameters)
    else:
        print 'Architecture NOT supported'


def blackbox(input_var,parameters):
    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.GlorotUniform()

    # set convolutional neural network
    network={}
    # input layer
    network["1"] = lasagne.layers.InputLayer(shape=(None,int(parameters['DS']['numChannels']), int(parameters['DS']['yInput']), int(parameters['DS']['xInput'])),input_var=input_var)
    # convolutional layer
    network["2"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=parameters['DL']['num_filters'], filter_size=parameters['DL']['filter_size'],nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    # pooling layer
    network["3"] = lasagne.layers.MaxPool2DLayer(network["2"], pool_size=parameters['DL']['pool_size'])
    # feed-forward layer
    network["4"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["3"], p=parameters['DL']['dropout_p']),num_units=parameters['DL']['num_dense_units'],nonlinearity=parameters['DL']['nonlinearity'])
    # output layer
    network["5"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["4"], p=parameters['DL']['dropout_p']),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    # returning the output layer standing for the net (network['5']), each layer separately (network) and the updated parameters for tracking.
    return network["5"],network,parameters

def timeFrequency(input_var,parameters):
    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.GlorotUniform()

    # set convolutional neural network
    network={}
    # input layer
    network["1"] = lasagne.layers.InputLayer(shape=(None,int(parameters['DS']['numChannels']), int(parameters['DS']['yInput']), int(parameters['DS']['xInput'])),input_var=input_var)
    # convolutional layer
    network["2"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=parameters['DL']['num_filters'], filter_size=parameters['DL']['filter_size'],nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    # pooling layer
    network["3"] = lasagne.layers.MaxPool2DLayer(network["2"], pool_size=parameters['DL']['pool_size'])
    # output layer
    network["4"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["3"], p=parameters['DL']['dropout_p']),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    # returning the output layer standing for the net (network['5']), each layer separately (network) and the updated parameters for tracking.
    return network["4"],network,parameters


def mergeTimeFrequency(input_var,parameters):
    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.GlorotUniform()

    # set convolutional neural network
    network={}
    # input layer
    network["1time"] = lasagne.layers.InputLayer(shape=(None,int(parameters['DS']['numChannels']), int(parameters['DS']['yInput']), int(parameters['DS']['xInput'])),input_var=input_var)
    # convolutional layer
    network["2time"] = lasagne.layers.Conv2DLayer(network["1time"], num_filters=parameters['DL']['num_filters'], filter_size=(1,60),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    # pooling layer
    network["3time"] = lasagne.layers.MaxPool2DLayer(network["2time"], pool_size=(40,1))

    # input layer
    network["1frequency"] = lasagne.layers.InputLayer(shape=(None,int(parameters['DS']['numChannels']), int(parameters['DS']['yInput']), int(parameters['DS']['xInput'])),input_var=input_var)
    # convolutional layer
    network["2frequency"] = lasagne.layers.Conv2DLayer(network["1frequency"], num_filters=parameters['DL']['num_filters'], filter_size=(32,1),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    # pooling layer
    network["3frequency"] = lasagne.layers.MaxPool2DLayer(network["2frequency"], pool_size=(1,80))
    network["3frequencyReshaped"]=lasagne.layers.ReshapeLayer(network["3frequency"],([0],[1],[3],[2]))

    network["3"]=lasagne.layers.ConcatLayer([network["3time"], network["3frequencyReshaped"]], axis=3, cropping=None)


    # feed-forward layer
    network["4"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["3"], p=parameters['DL']['dropout_p']),num_units=parameters['DL']['num_dense_units'],nonlinearity=parameters['DL']['nonlinearity'])
    # output layer
    network["5"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["4"], p=parameters['DL']['dropout_p']),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    # returning the output layer standing for the net (network['5']), each layer separately (network) and the updated parameters for tracking.
    return network["5"],network,parameters

def loadMergeTimeFrequency(input_var,parameters):
    parameters['DL']['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['DL']['W_init']=lasagne.init.GlorotUniform()

    # set convolutional neural network
    network={}
    # input layer
    network["1time"] = lasagne.layers.InputLayer(shape=(None,int(parameters['DS']['numChannels']), int(parameters['DS']['yInput']), int(parameters['DS']['xInput'])),input_var=input_var)
    # convolutional layer
    network["2time"] = lasagne.layers.Conv2DLayer(network["1time"], num_filters=parameters['DL']['num_filters'], filter_size=(1,60),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    # pooling layer
    network["3time"] = lasagne.layers.MaxPool2DLayer(network["2time"], pool_size=(40,1))
    # output layer
    network["4time"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["3time"], p=parameters['DL']['dropout_p']),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    # load best model
    import numpy as np

    name='./data/preloaded/Time_32_1-60_40-1'
    with np.load(name+'.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network["4time"], param_values)

    # input layer
    network["1frequency"] = lasagne.layers.InputLayer(shape=(None,int(parameters['DS']['numChannels']), int(parameters['DS']['yInput']), int(parameters['DS']['xInput'])),input_var=input_var)
    # convolutional layer
    network["2frequency"] = lasagne.layers.Conv2DLayer(network["1frequency"], num_filters=parameters['DL']['num_filters'], filter_size=(32,1),nonlinearity=parameters['DL']['nonlinearity'],W=parameters['DL']['W_init'])
    # pooling layer
    network["3frequency"] = lasagne.layers.MaxPool2DLayer(network["2frequency"], pool_size=(1,80))
    # output layer
    network["4frequency"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["3frequency"], p=parameters['DL']['dropout_p']),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    # load best model
    import numpy as np
    name='./data/preloaded/Frequency_32_32-1_1-80'
    with np.load(name+'.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network["4frequency"], param_values)

    # concatenate
    network["3frequencyReshaped"]=lasagne.layers.ReshapeLayer(network["3frequency"],([0],[1],[3],[2]))
    network["3"]=lasagne.layers.ConcatLayer([network["3time"], network["3frequencyReshaped"]], axis=3, cropping=None)
    # feed-forward layer
    network["4"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["3"], p=parameters['DL']['dropout_p']),num_units=parameters['DL']['num_dense_units'],nonlinearity=parameters['DL']['nonlinearity'])
    # output layer
    network["5"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["4"], p=parameters['DL']['dropout_p']),num_units=int(parameters['DS']['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    # returning the output layer standing for the net (network['5']), each layer separately (network) and the updated parameters for tracking.
    return network["5"],network,parameters
