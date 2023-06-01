# -*- coding: utf-8 -*-
"""
  Copyright 2022 Mitchell Isaac Parker

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import streamlit as st
from PIL import Image

from util.functions.path import get_file_path, get_dir_name, util_str, data_str

from util.pages.home_page import home_page
from util.pages.general_surfEP import general_surfEP
from util.pages.latent_varable_surfEP import latent_variable_surfEP



class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        img = Image.open(
            get_file_path(
                "web_image.png",
                dir_path=f"{get_dir_name(__file__)}/{util_str}/{data_str}",
            ),
        )

        st.set_page_config(page_title="SurfEp", page_icon=img, layout="wide")

        st.sidebar.markdown("## Main Menu")
        app = st.sidebar.selectbox(
            "Select Page", self.apps, format_func=lambda app: app["title"]
        )
        st.sidebar.markdown("---")
        app["function"]()


app = MultiApp()

app.add_app("Home Page", home_page)
app.add_app("SurfEP", general_surfEP)
app.add_app("Latent-variable SurfEP", latent_variable_surfEP)


app.run()
