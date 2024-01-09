# -*- coding: utf-8 -*-


import streamlit as st
from PIL import Image

from ..functions.table import mask_equal
from ..functions.col import pdb_code_col
from ..functions.path import pages_str, data_str, get_file_path
from ..functions.gui import load_st_table, write_st_end, create_st_button, show_st_structure, get_neighbor_path


def home_page():

    left_col, right_col = st.columns(2)

    df = load_st_table(__file__)

    img_1 = Image.open(
        get_file_path(
            "web_image.png",
            dir_path=get_neighbor_path(__file__, pages_str, data_str),
        )
    )

    left_col.image(img_1, output_format="PNG")
    # show_st_structure(mask_equal(df, pdb_code_col, "6oim"),
    #         zoom=1.2,
    #         width=400,
    #         height=300,
    #         cartoon_trans=0,
    #         surface_trans=1,
    #         spin_on=True,
    #         st_col=left_col)

    right_col.markdown("# SurfEp")
    right_col.markdown("### A tool for predicting alloy energetics")
    right_col.markdown("**Created by the Montemore group**")
    #right_col.markdown("**The Montemore group**")

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

   

    st.markdown("---")

    st.markdown(
        """
        ### Overview
        - Released by the [Montemore group at Tulane University](https://www.montemoregroup.org/). Email mmontemore@tulane.edu with questions or comments.
        - *SurfEP*, for Surface Energetics Prediction, allows the prediction of surface and bulk adsorption energies on metal alloy surfaces, as well as surface formation energies. These are currently implemented separately.
        - If you use this package or method in your research, please cite the relevant publication(s).

        
        """
    )

    left_col, right_col = st.columns(2)

    img_2 = Image.open(
        get_file_path(
            "TOC_homepage.png",
            dir_path=get_neighbor_path(__file__, pages_str, data_str),
        )
    )

    right_col.image(img_2, output_format="PNG")


    left_col.markdown(
        """
        ### Usage

        To the left, is a dropdown main menu for navigating to 
        each page:

        - **Home Page:** We are here!
        - **Surface adsorption energies:** Uses the SurfEp class for adsorption energy predictions.
        - **Bulk adsorption energies:** Uses the latent-variable SurfEp class (improved version of SurfEp) for adsorption energy predictions.
        - **Surface energies (stability):** Uses an element-centered fingerprint for surface stability predictions.
        
        """
    )
    st.markdown("---")

    left_info_col, right_info_col = st.columns(2)

    left_info_col.markdown(
        f"""
        The current version of this package is an early release. While it has been tested, it may have unexpected behavior in some situations.

        If comparing to your own DFT data, we suggest you do a linear fit between SurfEP predictions and your data, for the particular subset of alloys and adsorbate you're interested in.
        """
    )
    right_info_col.markdown(
        """
        Please feel free to contact us with any issues, comments, or questions.

        ##### Mattew Montemore 

        - Email:  <mmontemore@tulane.edu> 
    
        """,
        unsafe_allow_html=True,
    )


    write_st_end()
