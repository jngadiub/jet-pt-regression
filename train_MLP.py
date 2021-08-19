import os,sys

from optparse import OptionParser
import yaml

import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Model, Sequential, Input
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam, Nadam

from callbacks import all_callbacks

import models

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
 
def parse_config(config_file) :

    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

def get_pt_weights(options,inputArray,binning=5.):

	mybins = int(np.amax(inputArray)/binning)

	n, bins, patches = plt.hist(inputArray, range=(0.,np.amax(inputArray)), bins = mybins)
	n = np.array(n)
	print(n.shape)
	
	bins = np.array(bins)
	print(bins.shape)
	print(bins)	
	
	bins_lower = bins[:-1]
	bins_upper = bins[1:]	
	weights = np.nan_to_num(1./n)
	print(weights)

	# for each value find the correct bin
	w = []

	for i in range(len(inputArray)):
    	  above_x = bins_upper > inputArray[i]
    	  below_x = bins_lower <= inputArray[i]
    	  mask = above_x*below_x
    	  w_x = weights[np.argmax(mask)]
    	  w.append(w_x)
    	  #print(i,'x_value',x[i],'above_x',above_x,'below_x', below_x,'mask', mask,'w_x',w_x)
	
	w = np.array(w)
	print('weights shape:', w.shape)
	plt.figure()
	plt.hist(inputArray, range=(0.,np.amax(inputArray)), bins = mybins, weights=w)
	plt.xlabel('weighted inputs')
	plt.savefig(options.outputDir+'/debug_weights.png')
	return w

     
def plot_variables(dr,features_val,features,targets_val,reco_variables_val,l1_variables_val,reco_variables,options)   :

    for i in range(features_val.shape[1]):
       plt.figure()
       if features[i] != "phi" and features[i] != "eta": plt.semilogy()
       plt.hist(features_val[:,i], 100, facecolor='green', alpha=0.75)
       plt.xlabel('L1Jet_'+features[i])
       plt.savefig(options.outputDir+'/L1Jet_'+features[i]+'.png')

    for i in range(len(reco_variables)):
       plt.figure()
       if reco_variables[i] != "phi" and reco_variables[i] != "eta": plt.semilogy()
       #if reco_variables[i] == 'pT': plt.hist(reco_variables_val[:,i], bins=100, range=(0,500), facecolor='green', alpha=0.75)
       #else: plt.hist(reco_variables_val[:,i], 100, facecolor='green', alpha=0.75)
       plt.hist(reco_variables_val[:,i], 100, facecolor='green', alpha=0.75)
       plt.xlabel('RecoJet_'+reco_variables[i])
       plt.savefig(options.outputDir+'/RecoJet_'+reco_variables[i]+'.png')

    for i in range(len(reco_variables)):
       plt.figure()
       if reco_variables[i] == "phi" or reco_variables[i] == "eta": continue
       plt.semilogy()
       #if reco_variables[i] == 'pT': plt.hist(l1_variables_val[:,i], bins=200, range=(0,500), facecolor='green', alpha=0.75)
       #else: plt.hist(l1_variables_val[:,i], 100, facecolor='green', alpha=0.75)
       plt.hist(l1_variables_val[:,i], 100, facecolor='green', alpha=0.75)
       plt.xlabel('L1Jet_'+reco_variables[i])
       plt.savefig(options.outputDir+'/L1Jet_'+reco_variables[i]+'.png')
                  
    plt.figure()
    plt.semilogy()
    plt.hist(targets_val[:,0],120,range=(-2, 10),facecolor='green',alpha=0.75)   
    plt.xlabel('target')
    plt.savefig(options.outputDir+'/targets.png')

    plt.figure()
    plt.semilogy()
    plt.hist(dr[:,0],600,range=(0, 6),facecolor='green',alpha=0.75)   
    plt.xlabel('dR(muon,jet)')
    plt.savefig(options.outputDir+'/dr_L1muon_L1jet.png')
     
    plt.figure()
    plt.hist2d(reco_variables_val[:,0],targets_val[:,0],bins=[100,120],range=[[0, 500], [-2, 10]],norm=matplotlib.colors.LogNorm())
    plt.xlabel('Reco jet pT [GeV]')
    plt.ylabel('Target')
    plt.savefig(options.outputDir+'/target_vs_recoPT.png')
    
    plt.figure()
    plt.hist2d(l1_variables_val[:,0],targets_val[:,0],bins=[100,120],range=[[0, 500], [-2, 10]],norm=matplotlib.colors.LogNorm())
    plt.xlabel('L1 jet pT [GeV]')
    plt.ylabel('Target')
    plt.savefig(options.outputDir+'/target_vs_l1PT.png')    
    
    plt.figure()
    plt.hist2d(reco_variables_val[:,0],dr[:,0],bins=[100,400],range=[[0, 500], [0, 4]],norm=matplotlib.colors.LogNorm())
    plt.xlabel('Reco jet pT [GeV]')
    plt.ylabel('dR(muon,jet)')
    plt.savefig(options.outputDir+'/dr_vs_recoPT.png')      
    
    plt.figure()
    plt.hist2d(l1_variables_val[:,0],dr[:,0],bins=[100,400],range=[[0, 500], [0, 4]],norm=matplotlib.colors.LogNorm())
    plt.xlabel('L1 jet pT [GeV]')
    plt.ylabel('dR(muon,jet)')
    plt.savefig(options.outputDir+'/dr_vs_l1PT.png') 

    plt.figure()
    plt.hist2d(dr[:,0],targets_val[:,0],bins=[400,120],range=[[0, 4], [-2, 10]],norm=matplotlib.colors.LogNorm())
    plt.ylabel('Target')
    plt.xlabel('dR(muon,jet)')
    plt.savefig(options.outputDir+'/target_vs_dr.png') 
    

def get_features(options, yamlConfig):

    print
    
    h5File = h5py.File(options.inputFile,'r')
    eventTree = h5File['eventInfo'][()]
    eventNames = h5File['eventNames'][()]
    eventNames = [n.decode('utf-8') for n in eventNames]
    event_df = pd.DataFrame(eventTree,columns=eventNames)
    npvs = event_df['nPV'].values
    n_matched_jets = event_df['nMatchedJets'].values
    
    print("Event tree shape:",eventTree.shape)
    
    plt.figure()
    plt.hist(npvs,facecolor='green',alpha=0.75)   
    plt.xlabel('nPVs')
    plt.savefig(options.outputDir+'/npvs.png')    
    
    l1jetTree = h5File['l1Jet'][()]
    print("L1 jet tree shape:",l1jetTree.shape)    
    all_features = h5File['l1JetNames'][()]
    all_features = [f.decode('utf-8') for f in all_features]
    print("All inputs:")
    print(all_features)
     
    # List of features to use
    features = yamlConfig['Inputs']
    print("Inputs to use:")
    print(features)
        
    # Convert to dataframe
    features_df = pd.DataFrame(l1jetTree,columns=all_features)
    l1_ptcorr = features_df[["pT","eta","phi","index"]].values
    if features[0] == 'RawEt': features_df[["RawEt"]] = features_df[["RawEt"]].values/2. - features_df[["PUEt"]].values/2.
    features_df = features_df[features]

    #Prepare target values
    recojetTree = h5File['recoJet'][()]
    print
    print("Reco jet tree shape:",recojetTree.shape) 
    all_features = h5File['recoJetNames'][()]
    all_features = [f.decode('utf-8') for f in all_features]
    print("All reco variables:")
    print(all_features)

    #L1 Muons info
    l1muonTree = h5File['l1Muon'][()]
    print
    print("l1 muon tree shape:",l1muonTree.shape) 
    all_features_muons = h5File['l1MuonNames'][()]
    all_features_muons = [f.decode('utf-8') for f in all_features_muons]
    print("All l1 muon variables:")
    print(all_features_muons)
    muon_df = pd.DataFrame(l1muonTree,columns=all_features_muons)
    dr = muon_df[["dr"]].values

    #Reco Muons info
    rmuonTree = h5File['recoMuon'][()]
    print
    print("Reco muon tree shape:",rmuonTree.shape) 
    all_features_rmuons = h5File['recoMuonNames'][()]
    all_features_rmuons = [f.decode('utf-8') for f in all_features_rmuons]
    print("All reco muon variables:")
    print(all_features_rmuons)
    rmuon_df = pd.DataFrame(rmuonTree,columns=all_features_rmuons)
    dr_reco = rmuon_df[["dr"]].values
            
    print
    print("Regress to = ",yamlConfig['RegressTo'])
    
    targets_df = pd.DataFrame(recojetTree,columns=all_features)
    reco_variables_val = targets_df[["pT","eta","phi","index"]].values #we need these info for performance evalutaion
    if yamlConfig['RegressTo'] == 'ratio': targets_val = features_df[[features[0]]].values/targets_df[["pT"]].values #L1/reco
    elif yamlConfig['RegressTo'] == 'inv_ratio': targets_val = targets_df[["pT"]].values/features_df[[features[0]]].values #reco/L1
    elif yamlConfig['RegressTo'] == 'diff': targets_val = targets_df[["pT"]].values-features_df[[features[0]]].values #reco-L1
    elif yamlConfig['RegressTo'] == 'inv_diff': targets_val = features_df[[features[0]]].values-targets_df[["pT"]].values #L1-reco
    else: targets_val = targets_df[["pT"]].values
      
    # Convert to numpy array 
    features_val = features_df.values
    #targets_val = targets_df.values
    
    selections = yamlConfig['Selections']
    if 'LeadingJet' in selections:
    
     print("")
     print("Take only leading jets - new shapes:")
    
     reco_index = reco_variables_val[:,3]
     l1_index = l1_ptcorr[:,3]
    
     targets_val = targets_val[ np.where( (reco_index == 0) & (l1_index == 0) ),:]
     targets_val = targets_val[0]
     print(targets_val.shape)

     reco_variables_val = reco_variables_val[ np.where( (reco_index == 0) & (l1_index == 0) ),:]
     reco_variables_val = reco_variables_val[0]
     print(reco_variables_val.shape)
    
     features_val = features_val[ np.where( (reco_index == 0) & (l1_index == 0) ),:]
     features_val = features_val[0]
     print(features_val.shape)

     l1_ptcorr = l1_ptcorr[ np.where( (reco_index == 0) & (l1_index == 0) ),:]
     l1_ptcorr = l1_ptcorr[0]
     print(l1_ptcorr.shape)

     dr = dr[ np.where( (reco_index == 0) & (l1_index == 0) ),:]
     dr = dr[0]
     print(dr.shape)

    if 'RemoveSaturation' in selections:
    
     print("")
     print("Removing saturated jets at 1023.5 GeV - new shapes:")
     
     l1_pt = l1_ptcorr[:,0]
    
     targets_val = targets_val[np.where(l1_pt < 1023),:]
     targets_val = targets_val[0]
     print(targets_val.shape)

     reco_variables_val = reco_variables_val[np.where(l1_pt < 1023),:]
     reco_variables_val = reco_variables_val[0]
     print(reco_variables_val.shape)
    
     features_val = features_val[np.where(l1_pt < 1023),:]
     features_val = features_val[0]
     print(features_val.shape)
    
     l1_ptcorr = l1_ptcorr[np.where(l1_pt < 1023),:]
     l1_ptcorr = l1_ptcorr[0]
     print(l1_ptcorr.shape)

     dr = dr[np.where(l1_pt < 1023),:]
     dr = dr[0]
     print(dr.shape)

    if 'SaturationOnly' in selections:
    
     print("")
     print("Keep only saturated jets at 1023.5 GeV - new shapes:")
     
     l1_pt = l1_ptcorr[:,0]
    
     targets_val = targets_val[np.where(l1_pt > 1023),:]
     targets_val = targets_val[0]
     print(targets_val.shape)

     reco_variables_val = reco_variables_val[np.where(l1_pt > 1023),:]
     reco_variables_val = reco_variables_val[0]
     print(reco_variables_val.shape)
    
     features_val = features_val[np.where(l1_pt > 1023),:]
     features_val = features_val[0]
     print(features_val.shape)
    
     l1_ptcorr = l1_ptcorr[np.where(l1_pt > 1023),:]
     l1_ptcorr = l1_ptcorr[0]
     print(l1_ptcorr.shape)

     dr = dr[np.where(l1_pt > 1023),:]
     dr = dr[0]
     print(dr.shape)
     
    if 'RemoveMuons' in selections:
       
     print("")
     print("Removing jets overlapping with muons - new shapes:")
    
     targets_val = targets_val[np.where(dr > 0.5),:]
     targets_val = targets_val[0]
     print(targets_val.shape)

     reco_variables_val = reco_variables_val[np.where(dr > 0.5),:]
     reco_variables_val = reco_variables_val[0]
     print(reco_variables_val.shape)
    
     features_val = features_val[np.where(dr > 0.5),:]
     features_val = features_val[0]
     print(features_val.shape)
    
     l1_ptcorr = l1_ptcorr[np.where(dr > 0.5),:]
     l1_ptcorr = l1_ptcorr[0]
     print(l1_ptcorr.shape)

     dr = dr[np.where(dr > 0.5),:]
     dr = dr[0]
     print(dr.shape)
                            	
    #make plots of input features and targets
    plot_variables(dr,features_val,features,targets_val,reco_variables_val,l1_ptcorr,["pT","eta","phi","index"],options)
       
    njets = int(yamlConfig['Njets'])
    if njets !=-1:
     print
     print("Using only",njets,"jets in the sample")
     features_val = features_val[:njets,:]
     targets_val = targets_val[:njets,:]
     reco_variables_val = reco_variables_val[:njets,:]
     l1_ptcorr = l1_ptcorr[:njets]
     n_matched_jets = n_matched_jets[:njets]
            	 
    weights = get_pt_weights(options,features_val[:,0])
    
    #Split sample
    X_train_val, X_test, y_train_val, y_test, reco_variables_train, reco_variables_test, l1_ptcorr_train, l1_ptcorr_test, weights_train_val, weights_test = train_test_split(features_val, targets_val, reco_variables_val, l1_ptcorr, weights, test_size=0.2, random_state=42)
    
    if yamlConfig['NormalizeInputs']:
        print
        print("Normalize inputs")
        scaler = preprocessing.StandardScaler().fit(X_train_val)
        X_train_val = scaler.transform(X_train_val)
        X_test = scaler.transform(X_test)
     
    if options.split: 
     return X_train_val, X_test, y_train_val, y_test, reco_variables_train, reco_variables_test, l1_ptcorr_train, l1_ptcorr_test, weights_train_val, weights_test, n_matched_jets
    else:
     return features_val, targets_val, reco_variables_val, l1_ptcorr, weights, n_matched_jets, dr, dr_reco
  
if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-i','--input',action='store',type='string',dest='inputFile',default='L1Ntuple.h5', help='input h5 file')
    parser.add_option('-o','--output',action='store',type='string',dest='outputDir',default='train_MLP', help='output directory')
    parser.add_option('-c','--config',action='store',type='string', dest='config',default='train_MLP.yml', help='configuration file')
    parser.add_option("-s",'--split',action="store_true",dest="split", help="split in test and train",default=False)
    (options,args) = parser.parse_args()
     
    print("Input file:",options.inputFile)
    print("Outdir:",options.outputDir)

    if not options.split:
     print("ERROR: you are not splitting the dataset!! Rerun with option -s")
     sys.exit()
     
    if os.path.isdir(options.outputDir):
        input("Warning: output directory exists. Press Enter to continue...")
    else:
        os.mkdir(options.outputDir)        
    os.system('cp index.php %s'%options.outputDir)
    
    yamlConfig = parse_config(options.config)
    X_train_val, X_test, y_train_val, y_test, reco_variables_train, reco_variables_test, l1_ptcorr_train, l1_ptcorr_test, weights_train_val, weights_test, n_matched_jets  = get_features(options, yamlConfig)   
    
    print("****************************************")
    print("X train shape:",X_train_val.shape)
    print("X test shape:",X_test.shape)
    print("Y train shape:",y_train_val.shape)
    print("Y test shape:",y_test.shape)
    print("****************************************")
    
    print
    print("Model:",yamlConfig['KerasModel'])
    print("Loss function:",yamlConfig['Loss'])
    print("Loss function:",yamlConfig['Epochs'])

    model = getattr(models, yamlConfig['KerasModel'])  
    keras_model = model(Input(shape=X_train_val.shape[1:]))
    keras_model.summary()
           
    startlearningrate=0.0001
    adam = Adam(lr=startlearningrate)
    keras_model.compile(optimizer=adam, loss=yamlConfig['Loss'])

    
    callbacks=all_callbacks(stop_patience=1000, 
			lr_factor=0.5,
			lr_patience=10,
			lr_epsilon=0.000001, 
			lr_cooldown=2, 
			lr_minimum=0.0000001,
			outputDir=options.outputDir)

    history = keras_model.fit(X_train_val, y_train_val, batch_size = 1024, epochs = int(yamlConfig['Epochs']),
		validation_split = 0.25, shuffle = True, callbacks = callbacks.callbacks)#, sample_weight=weights_train_val)
		
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(options.outputDir+'/history.png')
