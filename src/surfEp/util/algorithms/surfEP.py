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
        self.dict_model_d, self.dict_model_p, self.dict_model_vad2, self.dict_model_ads = self.importAllModels(jsonDirectory,verbose)
        self.pureElementDataFrame = pd.read_csv(os.path.join(elementDataDirectory,'PureElementData.csv'),index_col=0)

    ## Next few functions deal with importing models
    def importAllModels(self,directory=None,verbose=True):
        '''Given the directory where the JSON files are, imports all models and returns dictionaries containing the models. '''

        import os
#        if directory = None:
#            directory = os.path.join(os.path.realpath(__file__),'JSONFiles')
        
        if verbose: print("Loading parameters from directory: ",directory)
        dict_model_d = {}
        for fileName in os.listdir(directory):
            if 'd_model' in fileName:
                if verbose: print(fileName)
                stringToDelete = '_d_model.txt'
                elementName = fileName.replace(stringToDelete,"")
                dict_model_d[elementName] = self.importLinearModel(directory+fileName)
        dict_model_p = {}
        for fileName in os.listdir(directory):
            if 'p_model' in fileName:
                if verbose: print(fileName)
                stringToDelete = '_p_model.txt'
                elementName = fileName.replace(stringToDelete,"")
                dict_model_p[elementName] = self.importLinearModel(directory+fileName)
        dict_model_vad2 = {}
        for fileName in os.listdir(directory):
            if 'vad2_model' in fileName:
                if verbose: print(fileName)
                stringToDelete = '_vad2_model.txt'
                elementName = fileName.replace(stringToDelete,"")
                dict_model_vad2[elementName] = self.importLinearModel(directory+fileName)

        dict_model_ads = {}
        for fileName in os.listdir(directory):
            if 'Ads_model' in fileName:
                if verbose: print(fileName)
                stringToDelete = '_Ads_model.txt'
                elementSite = fileName.replace(stringToDelete,"")
                element,site = elementSite.split('_')
                dict_model_ads[(element,site)] = self.importLinearModel(directory+fileName)

        return dict_model_d,dict_model_p,dict_model_vad2,dict_model_ads
    
    def importLinearModel(self, fileName):
        '''Imports linear model parameters into a scikitlearn model, from a json file'''
        import json
        from sklearn.linear_model import LinearRegression
        import numpy as np
        with open(fileName) as file:
            imported = json.loads(file.read())
        model = LinearRegression()
        model.coef_=np.array(imported['coefficients'])
        model.intercept_=imported['intercept']
        return model
    
    ## Next few functions deal with the actual adsorption energy prediction
    
    def atomsToAds(self, atomsList, adsorbateList, site, sitesIndicesList, surfaceIndicesList, returnDescriptions = False):
        '''For each given atoms and set of sites, returns a list of adsorption energies for each adsorbate in adsorbateList (so output is a list of lists of lists).
           Setting descriptionList to True makes the function return the composition, site, and adsorbate in a separate list sructured the same way as the adsorption energy prediction list.
        '''
        #Maybe TODO: take site name based on number of indices it has? This might be too clever.
        import numpy as np
        #Calc features
        sdCouplingList, dFillingDiffList, eCondDiffList = self.featuresForES(atomsList, surfaceIndicesList)
        #Calc ES
        predDCentList, predNumPList, predVad2fList, predVad2List = self.predictES(atomsList,surfaceIndicesList,sdCouplingList, dFillingDiffList, eCondDiffList, self.dict_model_d, self.dict_model_p, self.dict_model_vad2)

        predAdsList = []
        descriptList = []
        for atoms,sitesIndices,predDCent,predNumP,predVad2f,predVad2 in zip(atomsList,sitesIndicesList,predDCentList,predNumPList,predVad2fList,predVad2List):
            predAdsSurf = [ [] for site in adsorbateList]
            if returnDescriptions: descriptSurf = [ [] for site in adsorbateList]
            for siteIndices in sitesIndices: #For a given surface, there may be multiple sites
                predDCentSiteMean = np.mean(np.array(predDCent)[siteIndices])
                predNumPSiteMean = np.mean(np.array(predNumP)[siteIndices])
                predVad2SiteMean = np.mean(np.array(predVad2)[siteIndices])
                predVad2fSiteMean = np.mean(np.array(predVad2f)[siteIndices])
                esFeaturesAll = [predDCentSiteMean, predNumPSiteMean, predVad2fSiteMean, predVad2SiteMean]
                for i,adsorbate in enumerate(adsorbateList):
                    numberOfCoefficients = self.dict_model_ads[(adsorbate,site)].coef_
                    esFeatures = np.array(esFeaturesAll)[np.array(range(len(numberOfCoefficients)))]
                    predAdsSurf[i].append(self.ESToAds(adsorbate,site,esFeatures,self.dict_model_ads))
                    if returnDescriptions: descriptSurf[i].append(adsorbate+'/'+atoms.get_chemical_formula()+'_'+site)
            predAdsList.append(predAdsSurf)
            if returnDescriptions: descriptList.append(descriptSurf)
        if returnDescriptions:
            return predAdsList, descriptList
        else:
            return predAdsList
    
    def ESToAds(self, adsorbate,site,featureList,dict_model_ads):
        '''Simple wrapper. Species can be C, O, H, N, S, OH, K, maybe more. Site can be 'top', 'hollow', 'bridge'.
        featureList is in order: d-band center, number of p electrons, Vad^2f, Vad^2

        '''
    #     site = site.lower()
        model =  dict_model_ads[(adsorbate,site)]
        return model.predict([featureList])[0]
    
    
    def predictES(self, atomsList, surfaceIndicesList, sdCouplingList, dFillingDiffList, eCondDiffList, loaded_model_d, loaded_model_p, loaded_model_vad2):
        '''Given a set of atoms objects, the proper structural features, and the proper models, 
           predicts the relevant electronic structure properties.'''
        import numpy as np
        predDCentList = []
        predNumPList = []
        predVad2List = []
        predVad2fList = []
        for atoms,surfaceIndices,sdCoupling,dFillingDiff,eCondDiff in zip(atomsList,surfaceIndicesList, sdCouplingList, dFillingDiffList, eCondDiffList):
            predDCent = np.zeros(len(surfaceIndices))
            predNumP = np.zeros(len(surfaceIndices))
            predVad2 = np.zeros(len(surfaceIndices))
            for j,atom in enumerate(atoms[surfaceIndices]):
                num = surfaceIndices[j]
                featuresD = [np.array(dFillingDiff)[num],np.array(eCondDiff)[num],np.array(sdCoupling)[num]]
                featuresP = [np.array(sdCoupling)[num]]
                featuresvad2 = [np.array(dFillingDiff)[num]]
                predDCent[num] = loaded_model_d[atom.symbol].predict([featuresD])
                predNumP[num] = loaded_model_p[atom.symbol].predict([featuresP])
                predVad2[num] = loaded_model_vad2[atom.symbol].predict([featuresvad2])
            elements = np.array(atoms.get_chemical_symbols())[surfaceIndices]
            predVad2f = np.array([vad2*self.pureElementDataFrame.loc[element,'dFillingIdealized'] for element,vad2 in zip(elements,predVad2)])
            predDCentList.append(predDCent)
            predNumPList.append(predNumP)
            predVad2List.append(predVad2)
            predVad2fList.append(predVad2f)
        return predDCentList, predNumPList, predVad2fList, predVad2List
        
    def featuresForES(self, atomsList,surfaceIndicesList):
        import numpy as np
        '''Return order: sd coupling, delta f_d, delta electrical conductivity '''
        # Possible speedups: use numpy more; speed up calcPropertyDifference
        sdCouplingList = []
        dFillingDiffList = []
        eCondDiffList = []
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
            eCondDiff = np.array(eCondDiff)
            eCondDiffList.append(eCondDiff)

        return sdCouplingList, dFillingDiffList, eCondDiffList
    

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
