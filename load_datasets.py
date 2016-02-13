import os, pickle, urllib, json, sys
import numpy as np
from sklearn import preprocessing

from essentia.standard import *

def spectrogramClassification(parameters):

    #### FUNCTIONS ####

    def spect4song(dir,parameters):

        numSongs=0
        strA=[]
        for root, dirs, files in os.walk(dir):
            for d in dirs:
                for r, ds, fs in os.walk(root+'/'+d):
                    for f in fs:
                        numSongs=numSongs+1 # count number of songs in the dataset
                        strA.append(d) # load string annotations for each song
        parameters['DS']['numSongs']=numSongs
        dict={} # define a dictionary where each of the annotations correspond to an integer
        count=0
        for a in set(strA):
            dict[a]=count
            count=count+1
        parameters['DS']['dict']=dict
        #dict = {'ChaChaCha':0,'Samba':1,'Quickstep':2,'VienneseWaltz':3,'Tango':4,'Jive':5,'Waltz':6,'Rumba':7}
        parameters['DS']['numOutputNeurons'] = len(set(dict)) # number of output neurons equal to the number of classes

        # walk through the directory: compute spectrograms, normalize and annotate.
        annotation = np.zeros(parameters['DS']['numSongs'],dtype=np.uint8)+parameters['errorCode']
        songSpects = {}
        count=0
        for root, dirs, files in os.walk(dir):
            for d in dirs:
                for r, ds, fs in os.walk(root+'/'+d):
                    for f in fs:
                        print('    '+str(count)+'/'+str(parameters['DS']['numSongs'])+': '+root+'/'+d+'/'+f)
                        songSpects[count] = computeSpectrogram(root+'/'+d+'/'+f,parameters['DS']['frameSize'],parameters['DS']['hopSize'],parameters['DS']['windowType'],parameters['DS']['melBands'],parameters['DS']['specTransform'])
                        if parameters['DS']['inputNormWhere'] == 'local':
                            print('      LOCAL norm')
                            songSpects[count]=normalization(songSpects[count],parameters['DS']['inputNorm'])
                        annotation[count] = dict[d]
                        count = count+1

        # NORMALIZE GLOBALLY
        if parameters['DS']['inputNormWhere'] == 'global':
            numpy_songSpects, idcs = dict2numpy(songSpects)
            print('      GLOBAL norm')
            numpy_songSpects = normalization(numpy_songSpects,parameters['DS']['inputNorm'])
            ##!## for deployment, compute the mean and std of the training set and apply to test.
            songSpects = numpy2dict(numpy_songSpects,idcs)

        # Randomize dict!
        songSpects, annotation = shuffleData(songSpects, annotation)

        return songSpects, annotation # return onsets detections

    def splitTrainTest(songSpects, annotation, parameters):

        if parameters['folds']!=1: # k-fold cross-validation

            from sklearn.cross_validation import KFold
            kf = KFold(parameters['DS']['numSongs'], n_folds=parameters['folds'])
            for f, (train_index, test_index) in enumerate(kf):
                if f==parameters['currentFold']:
                    # print("TRAIN:", train_index, "\nTEST:", test_index)
                    annotation_train = annotation[train_index]
                    annotation_test = annotation[test_index]
                    songSpects_train_1 = songSpects.values()[:min(test_index)]
                    songSpects_train_2 = songSpects.values()[max(test_index)+1:]
                    songSpects_train = songSpects_train_1 + songSpects_train_2
                    songSpects_test = songSpects.values()[min(test_index):max(test_index)+1]

        else: # in casse is not cross validation, use pre-defined splits.

            cut_train=int(np.floor((parameters['trainSplit']+parameters['valSplit'])*parameters['DS']['numSongs']))
            songSpects_train = songSpects.values()[:cut_train]
            annotation_train = annotation[:cut_train]
            songSpects_test = songSpects.values()[cut_train:]
            annotation_test = annotation[cut_train:]

        return songSpects_train, annotation_train, songSpects_test, annotation_test

    def format4trainClass(songSpects,annotation,parameters):

        numSpects = 0
        for s in songSpects:
            for c in chunk(s,parameters['DS']['xInput']):
                numSpects = numSpects+1
        D = np.zeros(numSpects*parameters['DS']['yInput']*parameters['DS']['xInput'],dtype=np.float32).reshape(numSpects,parameters['DS']['numChannels'],parameters['DS']['yInput'],parameters['DS']['xInput'])
        A = np.zeros(numSpects,dtype=np.uint8)+parameters['errorCode']
        count = 0
        sCount = -1
        for s in songSpects:
            sCount=sCount+1
            #print '    '+str(sCount)+'/'+str(len(songSpects))
            for c in chunk(s,parameters['DS']['xInput']):
                # spectrogram chunks
                D[count][0] = c# normalization(c,'log') # NORMALIZE?!
                # associated annotation
                A[count] = annotation[sCount]
                count = count+1

        D,A = shuffleData(D, A)

        # split training/validation data
        if A.size ==0:
            cut_train=0
        else:
            cut_train=int(np.floor(parameters['trainSplit']*D.shape[0]/(parameters['trainSplit']+parameters['valSplit']))) ## ! ## this should be automatic when doing cross-validation
        X_train, X_val = D[:cut_train], D[cut_train+1:]
        y_train, y_val = A[:cut_train], A[cut_train+1:]

        return X_train, y_train, X_val, y_val

    def format4chunkTestClass(songSpects,annotation,parameters):

            numSpects = 0
            for s in songSpects:
                for c in chunk(s,parameters['DS']['xInput']):
                    numSpects = numSpects+1
            D = np.zeros(numSpects*parameters['DS']['yInput']*parameters['DS']['xInput'],dtype=np.float32).reshape(numSpects,parameters['DS']['numChannels'],parameters['DS']['yInput'],parameters['DS']['xInput'])
            A = np.zeros(numSpects,dtype=np.uint8)+parameters['errorCode']
            count = 0
            sCount = -1
            for s in songSpects:
                sCount=sCount+1
                #print '    '+str(sCount)+'/'+str(len(songSpects))
                for c in chunk(s,parameters['DS']['xInput']):
                    # spectrogram chunks
                    D[count][0] = c
                    # associated annotation
                    A[count] = annotation[sCount]
                    count = count+1

            X_test, y_test = shuffleData(D, A) ## ? ## is this necessary!?

            return X_test, y_test

    def computeSpectrogram(file,frameSize,hopSize,windowType,melBands,specTransform):
        'Compute spectrogram using Essentia python bindings'
        loader = essentia.standard.MonoLoader(filename = file)
        audio = loader()
        w = Windowing(type = windowType)
        spectrum = Spectrum()
        if specTransform=='mel':
            mel = MelBands(numberBands = melBands)

        spec = []
        for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
            if specTransform=='magnitudeSTFT':
                spec.append(spectrum(w(frame)))
            elif specTransform=='mel':
                spec.append(mel(spectrum(w(frame))))
        # we need to convert the list to an essentia.array first (== numpy.array of floats)
        spec = essentia.array(spec).T

        return spec

    #### RUN ####

    pickleFile = './data/preloaded/'+parameters['DS']['dataset']+'_'+str(parameters['DS']['frameSize'])+'_'+str(parameters['DS']['hopSize'])+'_'+str(parameters['DS']['specTransform'])+'_'+str(parameters['DS']['melBands'])+'_'+str(parameters['DS']['numChannels'])+'_'+str(parameters['DS']['windowType'])+'_'+str(parameters['DS']['yInput'])+'_'+str(parameters['DS']['xInput'])+'_'+str(parameters['DS']['inputNorm'])+'_'+str(parameters['DS']['inputNormWhere'])+'.pickle'
    # if it was already computed, simply load it.
    if os.path.exists(pickleFile):

        print "    Loading pre-computed spectrograms.."

        with open(pickleFile) as f:
            parameters_loaded = {}
            songSpects, annotation, parameters_loaded['DS'] = pickle.load(f)
            parameters['DS'] = parameters_loaded['DS']

    # otherwise, compute it!
    else:

        # define where the audios are
        root = './data/datasets/'
        dir = root+parameters['DS']['dataset']

        if not os.path.exists(dir) and parameters['DS']['dataset'] == 'genres':
            import urllib, tarfile
            source='http://opihi.cs.uvic.ca/sound/genres.tar.gz'
            print("Downloading %s" % source)
            urllib.urlretrieve(source, root+'genres.tar.gz')
            tfile = tarfile.open(root+'genres.tar.gz', 'r:gz')
            tfile.extractall(root)
        else:
            print 'Dataset not supported! You will have to implement it :)'

        # spect4song
        print '  Computing spectrograms..'
        songSpects, annotation = spect4song(dir,parameters)

        # Saving the loaded data
        with open(pickleFile, 'w') as f:
            pickle.dump([songSpects, annotation, parameters['DS']], f)

    # split data
    songSpects_train,annotation_train,songSpects_test,annotation_test = splitTrainTest(songSpects, annotation, parameters)

    # format4trainClass
    print '    Formatting training examples..'
    X_train, y_train, X_val, y_val = format4trainClass(songSpects_train,annotation_train,parameters)

    # format4testClass
    print '    Formatting testing examples..'
    X_test_utterances, y_test_utterances = format4chunkTestClass(songSpects_test,annotation_test,parameters)

    [X_test_majorityVote, y_test_majorityVote] = [songSpects_test,annotation_test]

    return X_train, y_train, X_val, y_val, X_test_utterances, y_test_utterances, X_test_majorityVote, y_test_majorityVote, parameters


#########
# UTILS #
#########
def dict2numpy(songSpects):
    length_songSpects = 0
    for s in songSpects:
        length_songSpects = length_songSpects + (songSpects[s].shape)[1]
    numpy_songSpects = np.zeros([(songSpects[s].shape)[0],length_songSpects],dtype=np.float32)
    idx=0
    idcs=[]
    for s in songSpects:
        idcs.append(idx)
        numpy_songSpects[:,idx:idx+songSpects[s].shape[1]] = songSpects[s]
        idx=idx+songSpects[s].shape[1]
    return numpy_songSpects, idcs

def numpy2dict(numpy_songSpects,idcs):
    dict_songSpects={}
    count=0
    for i in idcs:
        next=count+1
        if next >= len(idcs):
            dict_songSpects[count]=numpy_songSpects[:,idcs[count]:]
        else:
            dict_songSpects[count]=numpy_songSpects[:,idcs[count]:idcs[next]]
        count=count+1
    return dict_songSpects

def chunk(l, n):
    'Yield successive n-sized chunks from l.'
    out=[]
    for i in xrange(0, int(l.shape[1]/n)*n, n):
        out.append(l[:,i:i+n])
    return out

def normalization(data,inputNorm):
    'Normalize the data, choose a configuration.'
    if inputNorm=='log0m1v':
        data = np.log10(10000*data+1)
        data = (data-np.mean(data))/np.std(data)

    elif inputNorm=='None':
        print '  No normalization!'

    else:
        print '[ERROR!] This normalization does not exist!'

    return data

def shuffleData(spect,annotation):
    if type(spect)==type({}) and type(annotation)==type(np.zeros(1)): # if dict type!

        import random
        unordered = list(range(len(spect)))
        random.shuffle(unordered)
        ordered = list(range(len(spect)))
        spect_shuffled = {}
        annotation_shuffled = np.zeros(annotation.shape,dtype=np.uint8)+9999
        for o in ordered:
            spect_shuffled[o]=spect[unordered[o]]
            annotation_shuffled[o]=np.uint8(annotation[unordered[o]])
        print '      SHUFFLE: dict type'
        return spect_shuffled, annotation_shuffled

    elif type(spect)==type(np.zeros(1)): # if numpy type!

        assert len(spect) == len(annotation)
        p = np.random.permutation(len(spect))
        print '      SHUFFLE: numpy type'
        return spect[p], annotation[p]

    else:

        print '      CANNOT SHUFFLE THIS TYPE OF DATA!'

def load_dataset(parameters):
    'Choose which dataset you want to use and for which task - only supports classification, by now.'
    return spectrogramClassification(parameters)