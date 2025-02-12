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
#from ase.io import write


#import surfEP
import sys
# sys.path.append("../algorithms")
from ..algorithms.surfEP import surfEP

from ..functions.table import mask_equal
from ..functions.col import pdb_code_col
from ..functions.path import pages_str, data_str, get_file_path,get_dir_name,util_str
from ..functions.gui import load_st_table, write_st_end, create_st_button, show_st_structure, get_neighbor_path, show_st_fig


def general_surfEP():

    left_col, right_col = st.columns(2)

    df = load_st_table(__file__)


    img_1 = Image.open(
        get_file_path(
            "web_image.png",
            dir_path=get_neighbor_path(__file__, pages_str, data_str),
        )
    )

    # left_col.image(img_1, output_format="PNG")
   

    # right_col.markdown("# SurfEp")
    # right_col.markdown("### A tool for predicting alloy energetics")
    # right_col.markdown("**Created by the Montemore group**")
    

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

   


    #     create_st_button(link_text, link_url, st_col=st_col)

    #st.markdown("---")

    st.markdown(
        """
        ## SurfEP
        SurfEP is a general, efficient model that can be used to screen surface alloys for many reactions without 
        any quantum-based calculations. This model allows the prediction of the adsorption energies of a variety of
        species (explicitly shown for C, N, O, OH, H, S, K, F) on metal alloy surfaces that include combinations of nearly all of the d-block metals. 
        Please see the relevant publication [here](https://pubs.rsc.org/en/content/articlelanding/2020/cy/d0cy00682c)
        """
    )

    left_col, right_col = st.columns(2)

    

    left_col.markdown(
        """
        ### How does it work ?

        Complete all the parameters and the machine learning model will predict the adsorption energy of your chosen alloy

        """
    )
    img_1 = Image.open(
        get_file_path(
            "gen_screen.png",
            dir_path=get_neighbor_path(__file__, pages_str, data_str),
        )
    )

    left_col.image(img_1, output_format="PNG")

    left_col.markdown("")

    img_2 = Image.open(
        get_file_path(
            "doping_location.png",
            dir_path=get_neighbor_path(__file__, pages_str, data_str),
        )
    )

    left_col.image(img_2, output_format="PNG",)#width=100,caption="Doping location")

    with right_col:
        hostMetal=st.selectbox("Choose host metal",
                            ('Cu','Ag','Au','Ni','Pt','Pd','Co','Rh','Ir','Ru','Os','Re','Ti','Zr','Hf','Sc'))
        dopingMetal=st.selectbox("Select doping metal(s)",
                                    ('Cu','Ag','Au','Ni','Pt','Pd','Co','Rh','Ir','Fe','Ru','Os','Mn','Re','Cr','Mo','W','V','Ta','Ti','Zr','Hf','Sc'))
        adsorbate=st.selectbox("Choose host adsorbate",
                                ('C', 'N', 'O', 'OH', 'H', 'S', 'K', 'F'))
        dopingLocations=st.multiselect("Select doping locations",
                                       (i for i in range(18)))
        siteType=st.selectbox("Choose adsorbing site", 
                              ('Top','Bridge','Hollow'))
        
        site_dict={'Top':1,'Bridge':2,'FCCHollow':3,'HCPHollow':3, 'Hollow':3}  

        adsorptionSite_dict=st.multiselect("Select adsorbing sites indices",
                                       (i for i in range(9)), max_selections=site_dict[siteType])
        surfaceIndicesList = [[0,1,2,3,4,5,6,7,8]]

        ### Set up and view structure
        # Import host metal structure
        from pathlib import Path
        par_dir=Path(__file__).parent.parent
        json_path=get_file_path("datas/JSONFiles/",
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
                st.write(':red[This current version of SurfEp cannot predict ' + adsorbate + ' at the ' + siteType + ' site. Try another adsorbate and/or site]') 
                raise SystemExit
            try:
                st.write('Predicted adsorption energy (eV):', predAdsList[0][0][0])
                
            except IndexError:
                st.write(':red[Enter doping locations and/or adsorption site locations]') 
                raise SystemExit

            #add_adsorbate(slab,adsorbate,2.3,)
            def add_adsorbate():
                originalSlab = copy.deepcopy(slab)
                center = slab.get_scaled_positions()[adsorptionSite[0]]
                slab.wrap(center=center)
                adsPos = np.array(np.mean(slab.positions[adsorptionSite],axis=0))
                adsPos = adsPos + [0,0,0.75]
                adsAtoms = Atoms(adsorbate,positions=[adsPos]) 
                atomsNew = originalSlab+adsAtoms
                return atomsNew


            write(str(Path(__file__).parent) + '/del.png', add_adsorbate())#, rotation='10z,-80x')
            img_3 = Image.open(
            get_file_path(
            "del.png",
            dir_path=str(Path(__file__).parent),
            ))

            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")
            right_col.image(img_3, output_format="PNG")
        
    
    #st.markdown("---")


    write_st_end()
