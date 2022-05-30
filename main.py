import streamlit as st

from homepage import homepage 
from schemepage import schemepage

# if the counter is not in the session_state is the first time that the page is run
if 'a_counter' not in st.session_state:
        st.session_state['a_counter'] = 0

if 'email' not in st.session_state:
    st.session_state['email'] = "email"
if 'usage_num' not in st.session_state:
    st.session_state['usage_num'] = 0
if 'dimension' not in st.session_state:
    st.session_state['dimension'] = 0
if 'word_aggregation' not in st.session_state:
    st.session_state['word_aggregation'] = "word"
if 'graph_aggregation' not in st.session_state:
    st.session_state['graph_aggregation'] = "graph"



# if the counter is <= 0 run the homepage
if st.session_state['a_counter'] <= 0:
    homepage()
# if the counter is >= run the schemepage 
else: 
    st.set_page_config(layout="wide")
    schemepage(st.session_state['email'], st.session_state['usage_num'], st.session_state['dimension'], st.session_state['word_aggregation'], st.session_state['graph_aggregation'])

# Footer - it is shown in all the page
st.markdown("""---""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1")
    st.write('''Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.''')

with col2:
    st.subheader("2")
    st.write('''Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.''')

with col3:
    st.subheader("3") 
    st.write('''Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.''')   





