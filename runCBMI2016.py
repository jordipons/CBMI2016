import MIRdl

parameters = {}
parameters['DL']={}
parameters['DS']={}

# set architecture parameters

# # BLACK-BOX
# parameters['DL']['type'] = 'blackbox'
# parameters['DL']['filter_size']=(12,8) # (12,200)
# parameters['DL']['num_filters']=32
# parameters['DL']['pool_size']=(4,1)
# parameters['DL']['dropout_p']=.5
# parameters['DL']['num_dense_units']=200
# parameters['DS']['melBands'] = 40
# parameters['DS']['xInput'] = 80 # 250

# # TIME and FREQUENCY
# parameters['DL']['type'] = 'time' # 'time' 'frequency'
# parameters['DL']['filter_size']=(1,60) # TIME: (1,60) (1,200) | FREQUENCY: (30,1) (32,1) (34,1) (36,1) (38,1) (40,1)
# parameters['DL']['num_filters']=32
# parameters['DL']['pool_size']=(40,1) # TIME: (40,1) | FREQUENCY: (1,80)
# parameters['DL']['dropout_p']=.5
# parameters['DS']['melBands'] = 40
# parameters['DS']['xInput'] = 80  # TIME: 80 250 | FREQUENCY: 80

# # MERGE TIME-FREQUENCY: all filter sizes defined in buildArchitecture()
# parameters['DL']['type'] = 'mergeTimeFrequency'
# parameters['DL']['num_filters']=32
# parameters['DL']['dropout_p']=.5
# parameters['DL']['num_dense_units']=200
# parameters['DS']['melBands'] = 40
# parameters['DS']['xInput'] = 80

# TIME-FREQUENCYINIT (pre-initialization): all filter sizes defined in buildArchitecture()
parameters['DL']['type'] = 'loadMergeTimeFrequency' # 'cnn1', 'mlpAB'
parameters['DL']['num_filters']=32
parameters['DL']['dropout_p']=.5
parameters['DL']['num_dense_units']=200
parameters['DS']['melBands'] = 40
parameters['DS']['xInput'] = 80                     # options: frames


########################################
# the following parameters were fixed 
# during the different experiments:
########################################
# general parameters
parameters['errorCode'] = 999
parameters['testMethod'] = 'all'                    # options: 'majorityVote' or 'utterances' or 'all'
parameters['mode'] = 'train'     		    # options: 'train' or test. For test, introduce the mode(l) to be loaded. i.e. 'genres_cnn1_201814200862305619998386402939002111166'.
parameters['folds']=10                              # for k-fold cross-validation. if parameters['folds']==1: then the following pre-defined splits apply (meaning: NO cross-validation)
parameters['trainSplit'] = 0.8 
parameters['testSplit'] = 0.1        		    # 1 (using all dataset for testing)
parameters['valSplit'] = 1-parameters['trainSplit']-parameters['testSplit']
parameters['randomTry'] = 1

# Deep Learning parameters
parameters['DL']['num_epochs'] = 2000
parameters['DL']['batchSize'] = 10
parameters['DL']['lr'] = 0.01
parameters['DL']['momentum'] = 0
parameters['DL']['cost'] = 'crossentropy'           

# Data Set (input data) parameters
parameters['DS']['dataset'] = 'Ballroom'	    # options: 'Ballroom' or 'genres', for the GTZAN.
parameters['DS']['frameSize'] = 2048
parameters['DS']['hopSize'] = 1024
parameters['DS']['specTransform'] = 'mel'           # options: 'mel' and 'magnitudeSTFT'
parameters['DS']['numChannels'] = 1
parameters['DS']['windowType'] = 'blackmanharris62'
parameters['DS']['yInput'] = parameters['DS']['melBands']
parameters['DS']['inputNormWhere'] = 'global'
parameters['DS']['inputNorm'] = 'log0m1v'        # options: 'log0m1v' or 'None'

MIRdl.main(parameters)

# Output in ./data/results
