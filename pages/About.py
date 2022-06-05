import streamlit as st
from PIL import Image

# Log2Enc logo impagination
c1, c2, c3 = st.columns([1,4,1])
image = Image.open('./images/l2e.png')
c2.image(image)

st.write("""
        The first step is to upload an event log (it should contain a "label" attribute). The maximum allowed size is 15 MB, the only extension accepted is *.xes*.\n
        After uploading an event log, a sidebar appears on the left side of the screen. In the sidebar you can find different customization options:
        1. **Select the dimension**: this option allows you to choose the size of the resulting vectors. 
                                The minimum selectable size is 2, while the maximum size is 256.
                                Note that the minimum and maximum size for some algorithms is not the same. 
                                For BoostNE the minimum size is 17, the maximum size is 255. 
                                For GLEE, GraRep, Laplacian Eigenmaps and NetMF the maximum size is 16, while for HOPE the largest size reaches 32.
        2. **Select the aggregation for word embedding**: here you can choose the type of embedding between *Average* or *Max*;
        3. **Select the aggregation for graph embedding**: here you can choose whether to encode *Nodes* or *Edges*. 
                                If you choose to encode the *Nodes* you will need to specify whether to use *Max* or *Average* embedding.
                                If you choose to encode the *Edges* you will need to specify another type of embeddings.
        4. **Submit your email address**: The last step requires you to enter an email address. 
                                After three uses you will have to wait only 5 minutes to be able to upload an event log again.
            \n
        Press the Confirm button and wait for the processing to complete. Take it easy, it can take a few minutes. In the meantime you can do something else: as soon as the processing is finished you will receive an email notification!
        """)

    