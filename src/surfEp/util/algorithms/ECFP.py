import numpy as np
def findNeighborIndices(structure,nnDist,activeSpace):
    from ase.neighborlist import primitive_neighbor_list
    import numpy as np
    indicesOfNeighbors = [None] * len(activeSpace)
    nnListi, nnListj = primitive_neighbor_list('ij', structure.get_pbc(), structure.get_cell(), structure.get_positions(), nnDist)
    for i,atom in enumerate(activeSpace):
        whichToTake = np.where(nnListi==atom)[0]
        indicesOfNeighbors[i] = nnListj[whichToTake]
    return indicesOfNeighbors

def atomicNumberDict():
    return {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,
    'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,
    'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,
    'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,
    'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
    'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,
    'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,
    'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,
    'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,
    'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,
    'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,
    'Ac':89,'Th':90,'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,
    'Bk':97,'Cf':98,'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103,'Rf':104,
    'Db':105,'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112,
    'Nh':113,'Fl':114,'Mc':115,'Lv':116,'Ts':117,'Og':118}

def calcFingerprint(atoms,uniqueElements,maxNeighbors,atomPropDict,whichDistance=9,nnMult=1.1,repeat=None,sortVec=True,verbose=False):
    '''Given a single atoms object, calculates a structural fingerprint that may be useful for machine learning.'''
    import numpy as np
    if repeat is not None: atoms = atoms.repeat(repeat)
    nAtoms = atoms.get_global_number_of_atoms()
    if verbose: print(nAtoms)
    scalef = 1/100
    
    ### Find nearest neighbors of each atom, and their info
    nnDistance = np.sort(atoms.get_distances(1,list(range(nAtoms)),mic=True))[whichDistance]*nnMult
    neighborIndicesList = findNeighborIndices(atoms,nnDistance,list(range(nAtoms)))
    symbols = atoms.get_chemical_symbols()
    neighborSymbolsList = [np.array(symbols)[neighborIndices] for neighborIndices in neighborIndicesList ]

    chargeMult = []
    distancesList = []
    for i in range(nAtoms):
        tempChargeList = [atomPropDict[symbols[i]]*atomPropDict[neighborSymbol] for neighborSymbol in neighborSymbolsList[i]]
        chargeMult.append(np.array(tempChargeList))
        tempDistList = atoms.get_distances(i,neighborIndicesList[i],mic=True)
        distancesList.append(tempDistList)
        
    symbols = atoms.get_chemical_symbols()
    neighborSymbolsList = [np.array(symbols)[neighborIndices] for neighborIndices in neighborIndicesList ]
    
    coulombVecList = [chargeMult[i]/distancesList[i] for i in range(nAtoms)]
    coulombVecPaddedList = [np.pad(coulombVec,(0,maxNeighbors - coulombVec.shape[0])) for coulombVec in coulombVecList]

    if verbose: print('number of neighbors detected:', [coulombVec.shape[0] for coulombVec in coulombVecList])
    
    fullVector = []
    for elementAll in uniqueElements:
        coulombVecElementAll = []
        for i,elementStruc in enumerate(symbols):
            if elementStruc == elementAll:
                if sortVec:
                    coulombVecElementAll.append(np.sort(coulombVecPaddedList[i])[::-1])
                else:
                    coulombVecElementAll.append(coulombVecPaddedList[i])
        coulombVecElementComb = np.sum(coulombVecElementAll,axis=0) 
        if not isinstance(coulombVecElementComb,np.ndarray): coulombVecElementComb = np.zeros(maxNeighbors)
        if verbose: print(elementAll, np.array(coulombVecElementComb)*scalef*1/nAtoms)
        fullVector.extend(coulombVecElementComb)
        
    fullVector = np.array(fullVector)*scalef*1/nAtoms
        
    return fullVector
        
def calcAllFingerprints(atomsList, maxNeighbors, atomPropDict, repeat=None, sortVec=True, whichDistance=9, printFormula = False, verbose=False, nnMult=1.25):
    '''Wrapper function that calculates fingerprints for a list of atoms objects.'''
    import numpy as np
    uniqueElements = []
    for atoms in atomsList:
        uniqueElements = np.append(uniqueElements,np.unique(atoms.get_chemical_symbols()))
        uniqueElements = np.unique(uniqueElements)
    
    allFingerPrints = []
    for atoms in atomsList:
        if printFormula: print(atoms.get_chemical_formula())

        allFingerPrints.append(calcFingerprint(atoms,uniqueElements,maxNeighbors,atomicNumberDict(),verbose=verbose,whichDistance = whichDistance, nnMult=nnMult,repeat=repeat))
        
    return np.array(allFingerPrints)

