# -*- coding: utf-8 -*-


import streamlit as st
from PIL import Image

from ase.io import read, write
import numpy as np
from ase.visualize import view
import copy
import matplotlib.pyplot as plt
from ase.build import add_adsorbate
from ase import Atoms


#import surfEP
# import sys
# sys.path.append("../algorithms")
from ..algorithms.Lv_surfEP import surfEP

from ..functions.table import mask_equal
from ..functions.col import pdb_code_col
from ..functions.path import pages_str, data_str, get_file_path,get_dir_name,util_str
from ..functions.gui import load_st_table, write_st_end, create_st_button, show_st_structure, get_neighbor_path, show_st_fig


def latent_variable_surfEP():

    left_col, right_col = st.columns(2)

    df = load_st_table(__file__)


    # img_1 = Image.open(
    #     get_file_path(
    #         "web_image.png",
    #         dir_path=get_neighbor_path(__file__, pages_str, data_str),
    #     )
    # )

    # left_col.image(img_1, output_format="PNG")
   

    # right_col.markdown("# SurfEp")
    # right_col.markdown("### A tool for predicting alloy energetics")
    # right_col.markdown("**Created by the Montemore group**")
    # #right_col.markdown("**The Montemore group**")

    group_link_dict = {
        "Group's website": "https://www.montemoregroup.org/",
        "Repository": "https://bitbucket.org/mmmontemore/surfep/src/master/",
        
    }

    st.sidebar.markdown("## The Montemore Group")
    for link_text, link_url in group_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)

    paper_link_dict = {
        "SurfEP paper": "https://pubs.rsc.org/en/content/articlelanding/2020/cy/d0cy00682c",
        "Latent-variable SurfEP paper": "https://pubs.acs.org/doi/full/10.1021/jacsau.3c00419",
        "ECFP paper": "https://iopscience.iop.org/article/10.1088/2515-7655/aca122",
    }

    st.sidebar.markdown("## Research papers")
    for link_text, link_url in paper_link_dict.items():
        create_st_button(link_text, link_url, st_col=st.sidebar)

    

    #st.markdown("---")

    st.markdown(
        """
        ## Latent-Variable SurfEp
        Latent-Variable SurfEp is organized as a class, which is initialized by calling surfEP_lv(). There are optional arguments for the locations of the files that give the model parameters. Once a class object has been created, the primary function is atomsToAds(), which takes a list of atoms objects for the surfaces, a list of adsorbates, the site type, a list of the site indices (a list of lists, where each sublist contains the indices of a site), and a list of surface indices (the indices of all of the atoms in the surface). 

        - Possible host metals: ['Cu','Ag','Au','Ni','Pt','Pd','Co','Rh','Ir','Ru','Os','Re','Ti','Zr','Hf','Sc']
        - Possible dopant metals: ['Cu','Ag','Au','Ni','Pt','Pd','Co','Rh','Ir','Fe','Ru','Os','Mn','Re','Cr','Mo','W','V','Ta','Ti','Zr','Hf','Sc']
        - Possible adsorbates: Bulk --> ['C','O','H','N','CH','CH2','CH3','NH']  Surface --> ['C', 'N', 'O', 'OH', 'H', 'S', 'K', 'F']
        - Possible siteTypes: Bulk --> ['Top','Bridge','FCCHollow', 'HCPHollow'] Surface --> ['Top','Bridge','Hollow']
        - Not all siteTypes may be available for all species.
        - Other limitations: Currently, only bimetallic fcc(111) and hcp(0001) surfaces are supported, and each surface must be a pure metal in the bulk but can be doped with other atoms in the top two layers. Other geometries and architectures may run, but the predictions are unlikely to be reliable. Further, if the surface dramatically reconstructs or the adsorbate relaxes out of site, the predictions are unlikely to be reliable. The surfaceIndicesList must be sequential and start at 0 (hence, the first few atoms in the atoms object must be in the top layer of the slab).
        
        """
    )

    left_col, right_col = st.columns(2)

    left_col.markdown(
        """
        ### How does it work ?

        Complete all the parameters and the machine learning model will predict the adsorption energy of your chosen alloy

        """
    )
    img_2 = Image.open(
        get_file_path(
            "latent_variable.png",
            dir_path=get_neighbor_path(__file__, pages_str, data_str),
        )
    )

    left_col.image(img_2, output_format="PNG")
    left_col.markdown("")

    img_3 = Image.open(
        get_file_path(
            "doping_location.png",
            dir_path=get_neighbor_path(__file__, pages_str, data_str),
        )
    )

    left_col.image(img_3, output_format="PNG",)#width=100,caption="Doping location")

    with right_col:
        
        pred_type=st.radio(" Prediction type :",
                 ('Fit to Bulk alloys','Fit to Surface alloys'))
        
        st.markdown("---")
        
        hostMetal=st.selectbox("Choose host metal",
                            ('Cu','Ag','Au','Ni','Pt','Pd','Co','Rh','Ir','Ru','Os','Re','Ti','Zr','Hf','Sc'))
        dopingMetal=st.selectbox("Select doping metal(s)",
                                    ('Cu','Ag','Au','Ni','Pt','Pd','Co','Rh','Ir','Fe','Ru','Os','Mn','Re','Cr','Mo','W','V','Ta','Ti','Zr','Hf','Sc'))
        if pred_type=='Fit to Surface alloys':
            adsorbate=st.selectbox("Choose host adsorbate",
                                    ('C', 'N', 'O', 'OH', 'H', 'S', 'K', 'F'))
        elif pred_type=='Fit to Bulk alloys':
            adsorbate=st.selectbox("Choose host adsorbate",
                                    ('C','O','H','N','CH','CH2','CH3','NH'))
        dopingLocations=st.multiselect("Select doping locations",
                                       (i for i in range(18)))
        if pred_type=='Fit to Surface alloys':
            siteType=st.selectbox("Choose adsorbing site", 
                                ('Top','Bridge','Hollow'))
        elif pred_type=='Fit to Bulk alloys':
            siteType=st.selectbox("Choose adsorbing site", 
                                ('Top','Bridge','FCCHollow','HCPHollow'))
            
        site_dict={'Top':1,'Bridge':2,'FCCHollow':3,'HCPHollow':3, 'Hollow':3}  

        adsorptionSite_dict=st.multiselect("Select adsorbing sites indices",
                                       (i for i in range(9)), max_selections=site_dict[siteType])
        surfaceIndicesList = [[0,1,2,3,4,5,6,7,8]]

        

        ### Set up and view structure
        # Import host metal structure
        from pathlib import Path
        par_dir=Path(__file__).parent.parent

        if pred_type=='Fit to Bulk alloys':
            json_path=get_file_path("datas/JSONFiles_Lv_bulk/",
                    dir_path=str(par_dir)#f"{get_dir_name(__file__)}/{util_str}",
                    )
        elif pred_type=='Fit to Surface alloys':
            json_path=get_file_path("datas/JSONFiles_Lv_surface/",
                    dir_path=str(par_dir)#f"{get_dir_name(__file__)}/{util_str}",
                    )
            
        element_path=get_file_path("datas/",
                dir_path=str(par_dir)#f"{get_dir_name(__file__)}/{util_str}",
                )
        if st.button('Predict'):

            slab = read(par_dir.joinpath('./datas/HostStructures/POSCAR_'+hostMetal))
            # Dope in metal
            symbols = np.array(slab.get_chemical_symbols())
            symbols[dopingLocations] = dopingMetal
            slab.set_chemical_symbols(symbols)

            adsorptionSite=list(adsorptionSite_dict)

            adsPredictor = surfEP(verbose=False,jsonDirectory =json_path,elementDataDirectory=element_path)
            try:
                predAdsList = adsPredictor.atomsToAds([slab] ,[adsorbate],siteType,[adsorptionSite], surfaceIndicesList)
            except KeyError:
                st.write(':red[This current version of Latent-variable SurfEp cannot predict ' + adsorbate + ' at the ' + siteType + ' site. Try another adsorbate and/or site]') 
                raise SystemExit
            
            #st.write('Predicted adsorption energy (eV):', predAdsList[0][0][0])

            #temporary fix (sign of mamums dataset has been set to negative from the source code), i'm now fixing the sign for montemore's dataset here
            
            try:
                if pred_type=='Fit to Bulk alloys':
                    st.write('Predicted adsorption energy (eV):', predAdsList[0][0][0])
            

                elif pred_type=='Fit to Surface alloys':
                    st.write('Predicted adsorption energy (eV):', -1 * predAdsList[0][0][0])
                    
            except IndexError:
                st.write(':red[Enter doping locations and/or adsorption site locations]') 
                raise SystemExit
                
            def add_adsorbate():
                originalSlab = copy.deepcopy(slab)
                center = slab.get_scaled_positions()[adsorptionSite[0]]
                slab.wrap(center=center)
                adsPos = np.array(np.mean(slab.positions[adsorptionSite],axis=0))
                adsPos = adsPos + [0,0,0.75]
                adsAtoms = Atoms(adsorbate,positions=[adsPos]) 
                atomsNew = originalSlab+adsAtoms
                return atomsNew

            write(str(Path(__file__).parent) + '/del.png', add_adsorbate(),)# rotation='10z,-80x')
            img_4 = Image.open(
            get_file_path(
            "del.png",
            dir_path=str(Path(__file__).parent),
            ))

            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")
            right_col.image(img_4, output_format="PNG")
        
    
    #st.markdown("---")

    write_st_end()


    raise SystemExit
   
