class surfEP():
    def __init__(self, jsonDirectory = None, elementDataDirectory = None, verbose=True):
        import pandas as pd
        import os
        #TODO: check if any files exist. If not, give helpful error message.
        if jsonDirectory == None:
            jsonDirectory = os.path.join(os.path.dirname(os.path.realpath(__file__)),'JSONFiles/')
        if verbose: print("Loading parameters from directory: ", jsonDirectory)
        if elementDataDirectory == None:
            elementDataDirectory = os.path.dirname(os.path.realpath(__file__))
        if verbose: print("Loading element data from directory: ", elementDataDirectory)
        self.dict_host_coef, self.dict_guest_models = self.importAllModels(jsonDirectory,verbose)
        self.pureElementDataFrame = pd.read_csv(os.path.join(elementDataDirectory,'PureElementData.csv'),index_col=0)

    ## Next few functions deal with importing models
    def importAllModels(self,directory=None,verbose=True):
        '''Given the directory where the JSON files are, imports all models and returns dictionaries containing the models. '''
        import json
        import os
#        if directory = None:
#            directory = os.path.join(os.path.realpath(__file__),'JSONFiles')
        
        if verbose: print("Loading parameters from directory: ",directory)
        dict_host_coef = {}
        for fileName in os.listdir(directory):
            if 'hostParams' in fileName:
                if verbose: print(fileName)
                stringToDelete = 'hostParams_'
                elementName = fileName.replace(stringToDelete,"").replace(".txt","")
                with open(directory+fileName) as file:
                    imported = json.loads(file.read())
                dict_host_coef[elementName] = imported
        dict_guest_models = {}
        for fileName in os.listdir(directory):
            if 'guestParams' in fileName:
                if verbose: print(fileName)
                stringToDelete = 'guestParams__'
                name = fileName.replace(stringToDelete,"").replace(".txt.","")
                split = name.split('__')
                guestName = split[0]
                siteName = split[1]
                dict_guest_models[(guestName,siteName)] = self.importLinearModel(directory+fileName)
        return dict_host_coef,dict_guest_models
    
    def importLinearModel(self, fileName):
        '''Imports linear model parameters into a scikitlearn model, from a json file'''
        import json
        from sklearn.linear_model import LinearRegression
        import numpy as np
        with open(fileName) as file:
            imported = json.loads(file.read())
        model = LinearRegression(fit_intercept=False)
        model.coef_=np.array(imported['coefficients'])
        model.intercept_= imported['intercept'] ### TODO: Not sure I understand how this intercept is working, but seems to be necessary.
        return model
    
    ## Next few functions deal with the actual adsorption energy prediction
    
    def atomsToAds(self, atomsList, adsorbateList, site, sitesIndicesList, surfaceIndicesList, returnDescriptions = False, verbose = False):
        '''For each given atoms and set of sites, returns a list of adsorption energies for each adsorbate in adsorbateList (so output is a list of lists of lists).
           Setting descriptionList to True makes the function return the composition, site, and adsorbate in a separate list sructured the same way as the adsorption energy prediction list.
        '''
        #Maybe TODO: take site name based on number of indices it has? This might be too clever.
        import numpy as np
        ###Calc features
        sdCouplingList, dFillingDiffList, eCondDiffList, dFillingMultList = self.calcFeatures(atomsList,surfaceIndicesList)
        ###Calc latent
        sdLatentList, dFillingLatentList, eCondLatentList, dFillingMultLatentList, hostConstantsList = self.featuresToLatent(atomsList,surfaceIndicesList,sdCouplingList, dFillingDiffList, eCondDiffList, dFillingMultList)

        predAdsList = []
        descriptList = []
        ### NEED TO FIX THIS:
        for atoms, sitesIndices, sdLatent, dFillingLatent, eCondLatent, dFillingMultLatent, hostConstants in zip(atomsList, sitesIndicesList, sdLatentList, dFillingLatentList, eCondLatentList, dFillingMultLatentList, hostConstantsList):
            predAdsSurf = [ [] for site in adsorbateList]
            if returnDescriptions: descriptSurf = [ [] for site in adsorbateList]
            for siteIndices in sitesIndices: #For a given surface, there may be multiple sites
                sdCouplingSiteMean = np.mean(np.array(sdLatent)[siteIndices])
                dFillingSiteMean = np.mean(np.array(dFillingLatent)[siteIndices])
                eCondSiteMean = np.mean(np.array(eCondLatent)[siteIndices])
                dFillingMultSiteMean = np.mean(np.array(dFillingMultLatent)[siteIndices])
                latentFeatures = [sdCouplingSiteMean, dFillingSiteMean, eCondSiteMean, dFillingMultSiteMean]
                hostConstantsSiteMean = np.mean(np.array(hostConstants)[siteIndices])
                for i,adsorbate in enumerate(adsorbateList):
#                     numberOfCoefficients = self.dict_model_ads[(adsorbate,site)].coef_
#                     esFeatures = np.array(latentFeaturesAll)[np.array(range(len(numberOfCoefficients)))]
                    if verbose:
                        print('latent variables:', latentFeatures, hostConstantsSiteMean)
                    predAdsSurf[i].append(self.latentToAds(adsorbate, site, latentFeatures, hostConstantsSiteMean))
                    if returnDescriptions: descriptSurf[i].append(adsorbate+'/'+atoms.get_chemical_formula()+'_'+site)
            predAdsList.append(predAdsSurf)
            if returnDescriptions: descriptList.append(descriptSurf)
        if returnDescriptions:
            return predAdsList, descriptList
        else:
            return predAdsList
    
    def latentToAds(self, adsorbate, site, featureList, hostConstant, dict_guest_models = None):
        '''Simple wrapper. Species can be C, O, H, N, S, OH, K, maybe more. Site can be 'top', 'hollow', 'bridge'.
        featureList is in order: 'sd_coupling', 'd_filling_n', 'e_conductivity_n', 'd_filling_mult.'

        '''
        ### TODO: Check whether I'm correctly using hostConstant. Probably just take it as an argument, and add it here.
    #     site = site.lower()
        import numpy as np
        if dict_guest_models == None: dict_guest_models = self.dict_guest_models
        model =  dict_guest_models[(adsorbate,site)]
        return np.negative(model.predict([featureList])[0] + hostConstant)#added np.negative here to make predictions consistent with previous SurfEP
    
    
    def featuresToLatent(self, atomsList, surfaceIndicesList, sdCouplingList, dFillingDiffList, eCondDiffList, dFillingMultList, dict_host_coef=None):
        '''Given a set of atoms objects, the proper structural features, and the proper models, 
           predicts the relevant electronic structure properties.'''
        ### TODO: add constant.
        import numpy as np
        if dict_host_coef is None: dict_host_coef = self.dict_host_coef
        sdLatentList = []
        dFillingLatentList = []
        eCondLatentList = []
        dFillingMultLatentList = []
        hostConstantsList = []
        for atoms,surfaceIndices,sdCoupling,dFillingDiff,eCondDiff,dFillingMult in zip(atomsList,surfaceIndicesList, sdCouplingList, dFillingDiffList, eCondDiffList, dFillingMultList):
            sdLatent = np.zeros(len(surfaceIndices))
            dFillingLatent = np.zeros(len(surfaceIndices))
            eCondLatent = np.zeros(len(surfaceIndices))
            dFillingMultLatent = np.zeros(len(surfaceIndices))
            hostConstants = np.zeros(len(surfaceIndices))
            ### This loop should work even if the surfaceIndices are in a weird order, but the rest of the code may not.
            for j,atom in enumerate(atoms[surfaceIndices]):
                num = surfaceIndices[j]
#                 features_sd = np.array([np.array(sdCoupling)[num]])
#                 print(features_sd)
#                 features_dFill = np.array([np.array(dFillingDiff)[num]])
#                 features_eCond = np.array([np.array(eCondDiff)[num]])
#                 features_dFillMult = np.array([np.array(dFillingMult)[num]])

#                 sdLatent[num] = dict_host_coef[atom.symbol]['sd_coupling']*[features_sd]
#                 dFillingLatent[num] = dict_host_coef[atom.symbol]['d_filling_n']*[features_dFill]
#                 eCondLatent[num] = dict_host_coef[atom.symbol]['e_conductivity_n']*[features_eCond]
#                 dFillingMultLatent[num] = dict_host_coef[atom.symbol]['d_filling_mult']*[features_dFill]
#                 hostConstants[num] = dict_host_coef[atom.symbol]


                sdLatent[num] = dict_host_coef[atom.symbol]['sd_coupling']*np.array(sdCoupling)[num]
                dFillingLatent[num] = dict_host_coef[atom.symbol]['d_filling_n']*np.array(dFillingDiff)[num]
                eCondLatent[num] = dict_host_coef[atom.symbol]['e_conductivity_n']*np.array(eCondDiff)[num]
                dFillingMultLatent[num] = dict_host_coef[atom.symbol]['d_filling_mult']*np.array(dFillingMult)[num]
                hostConstants[num] = dict_host_coef[atom.symbol]['constant']

#                 sdLatent[j] = dict_host_coef[atom.symbol]['sd_coupling']*np.array(sdCoupling)[j]
#                 dFillingLatent[j] = dict_host_coef[atom.symbol]['d_filling_n']*np.array(dFillingDiff)[j]
#                 eCondLatent[j] = dict_host_coef[atom.symbol]['e_conductivity_n']*np.array(eCondDiff)[j]
#                 dFillingMultLatent[j] = dict_host_coef[atom.symbol]['d_filling_mult']*np.array(dFillingMult)[j]
#                 hostConstants[j] = dict_host_coef[atom.symbol]['constant']
            sdLatentList.append(sdLatent)
            dFillingLatentList.append(dFillingLatent)
            eCondLatentList.append(eCondLatent)
            dFillingMultLatentList.append(dFillingMultLatent)
            hostConstantsList.append(hostConstants)
        return sdLatentList, dFillingLatentList, eCondLatentList, dFillingMultLatentList, hostConstantsList
        
    def calcFeatures(self, atomsList,surfaceIndicesList):
        import numpy as np
        '''Return order: sd coupling, delta f_d, delta electrical conductivity '''
        scaleECond = 10**-7 ###Puts conductivity on a more similar scale to other params.
        # Possible speedups: use numpy more; speed up calcPropertyDifference
        sdCouplingList = []
        dFillingDiffList = []
        eCondDiffList = []
        dFillingMultList = []
        for atoms,surfaceIndices in zip(atomsList,surfaceIndicesList):
            centralAtom, atomsFirstShell = self.findNeighborAtoms(atoms,surfaceIndices)
            sdCoupling = [] 
            for whichAtom in range(len(surfaceIndices)):
                rdExp = 1.5 #s-d coupling should be 1.5
                distExp = 3.5 #s-d coupling should be 3.5
                couplingTemp = self.calculateCoupling(rdExp=rdExp,distExp=distExp,centralAtom=centralAtom[whichAtom],shellAtoms=atomsFirstShell[whichAtom],pureElementDataFrame=self.pureElementDataFrame)
                sdCoupling.append(couplingTemp)
            sdCouplingList.append(sdCoupling)

            name = "dFillingIdealized"
            dFillingDiff = []
            for whichAtom in range(len(surfaceIndices)):
                dFillingDiff.append(self.calcPropertyDifference(name,centralAtom[whichAtom],atomsFirstShell[whichAtom],self.pureElementDataFrame))
            dFillingDiff = np.array(dFillingDiff)
            dFillingDiffList.append(dFillingDiff)

            name = "ElectricalConductivity"
            eCondDiff = []
            for whichAtom in range(len(surfaceIndices)):
                eCondDiff.append(self.calcPropertyDifference(name,centralAtom[whichAtom],atomsFirstShell[whichAtom],self.pureElementDataFrame))
            eCondDiff = np.array(eCondDiff)*scaleECond
            eCondDiffList.append(eCondDiff)
            
            surfaceElements = np.array(atoms.get_chemical_symbols())[surfaceIndices]
            dFillingSurf = np.array([self.pureElementDataFrame.loc[element,'dFillingIdealized'] for element in surfaceElements])
            dFillingMult = dFillingDiff*dFillingSurf
            dFillingMultList.append(dFillingMult)

        return sdCouplingList, dFillingDiffList, eCondDiffList, dFillingMultList
    

    def findNeighborAtoms(self, atoms,indices,whichAtomDist=0):
        import numpy as np
        from ase import Atoms
        '''Given an atoms object and a list of indices, 
        returns lists of (1) an atoms object with the central atom, and (2) an atoms object containing the first neighbors of each atom in the indices list.'''
        whichAtomDist = 0 # Just used in distances calculation. Shouldn't matter unless not a perfect lattice.
        distances = np.unique(np.round(atoms.get_distances(whichAtomDist,range(len(atoms)),mic=True),decimals=4)) #Unique, sorted distances.
        atomsFirstShellList = []
        centralAtomList = []
        firstNeighborIndicesList = self.findNeighborIndices(atoms,distances[1]*1.1,indices) #Can I move this out of the loop?
        for whichAtom,firstNeighborIndices in zip(indices,firstNeighborIndicesList):
            centralAtomList.append(atoms[whichAtom])
            atomsFirstShell = Atoms(cell=atoms.cell,pbc=True)
            for index in firstNeighborIndices:
                atomsFirstShell.append(atoms[index])
            atomsFirstShellList.append(atomsFirstShell)
        return centralAtomList, atomsFirstShellList
    
    def findNeighborIndices(self,structure,nnDist,activeSpace):
        from ase.neighborlist import primitive_neighbor_list
        import numpy as np
        indicesOfNeighbors = [] #[None] * len(activeSpace)
        nnListi, nnListj = primitive_neighbor_list('ij', structure.get_pbc(), structure.get_cell(), structure.get_positions(), nnDist)
        for i,atom in enumerate(activeSpace):
            whichToTake = np.where(nnListi==atom)[0]
            indicesOfNeighbors.append(nnListj[whichToTake])
        return indicesOfNeighbors 
    
    def calculateCoupling(self,rdExp,distExp,centralAtom,shellAtoms,pureElementDataFrame):
        import numpy as np
        import copy
        centralAndShell = copy.deepcopy(shellAtoms)
        centralAndShell.append(centralAtom)
        rd_nnList = np.array([pureElementDataFrame.loc[element,'rd'] for element in shellAtoms.get_chemical_symbols()])
        rd_ca = pureElementDataFrame.loc[centralAtom.symbol,"rd"]
        distList = centralAndShell.get_distances(-1,range(len(centralAndShell)-1),mic=True)
        return np.sum(rd_ca**(rdExp/2)*rd_nnList**(rdExp/2)/(distList)**distExp)
    
    def calcPropertyDifference(self, propertyName,centralAtom,shellAtoms,pureElementDataFrame):
        import numpy as np
        import copy
        centralAndFirstShell = copy.deepcopy(shellAtoms)
        centralAndFirstShell.append(centralAtom)
        property_nnList = np.array([pureElementDataFrame.loc[element,propertyName] for element in shellAtoms.get_chemical_symbols()])
        property_ca = pureElementDataFrame.loc[centralAtom.symbol,propertyName]
        return np.mean(property_ca-property_nnList)
