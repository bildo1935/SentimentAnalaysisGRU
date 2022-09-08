import streamlit as st
import subprocess
import sys

st.title('Welcome to AI4Good.')

st.header('Introduction.')
st.markdown(f'''
Recent volatility and turbulence has caused a marked increase in distress of the Singaporean population. 
It has been found that many who suffer from said mental distress attempt to find relief through (anonymous) participation  
in online communities e.g. Reddit.
''')

st.header('How it works.')
st.markdown(f'''This project mines three subreddits (r/Singapore, r/NationalServiceSG and r/SingaporeRaw) for posts as well as their authors. 
It preprocesses the text and transforms it via Word2Vec embedddings.
It is then fed as input into a trained Recurrent Neural Network with Gated Recurrent Units. 
The output is in the form of two classes (0 or 1 each representing mental stability and a relative lack thereof respectively).
Authors whose texts are classified as representing psychological instability are flagged up.''')

st.markdown(f'''The dataset used for training is a modified version of the training data used in  
[On the State of Social Media Data for Mental Health Research](https://arxiv.org/abs/2011.05233).  
More of such datasets can be found [here](https://github.com/kharrigian/mental-health-datasets).
''')

roller = st.button('Execute Model.')
if roller:
    st.write('This might take a while.')
    result = subprocess.run([sys.executable, "-c", "project.py"], capture_output=True, text=True)
    print(result.stdout)

