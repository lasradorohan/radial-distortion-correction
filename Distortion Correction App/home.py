import streamlit as st
import os
import io
import PIL
import time
import shutil
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import glob

import distorter
import autodetector



MENU = {
    "Auto-Detect and Correct Images": 1,
    "Distort Images": 2
}

def main():
    st.sidebar.title("Function Selector")
    # menu_sel = st.sidebar.radio("Select:", list(MENU.keys()))
    menu_sel = st.sidebar.selectbox("Select:", list(MENU.keys()))

    if MENU[menu_sel] == 1:
        autodetector.main()
    if MENU[menu_sel] == 2:
        distorter.main()
    
    # menu_opt = MENU[menu_sel]
    # with st.spinner(f'Loading {menu_sel}...'):
    #     menu_opt.main()


if __name__ == "__main__":
    main()
