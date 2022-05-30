from datetime import *
import streamlit as st

timestamp = datetime.now()
st.write("datetime now :", timestamp)

iso = timestamp.isoformat()
st.write("datetime now isoformat: ", iso)

end = datetime.fromisoformat(iso) + timedelta(hours=2)
st.write("datetime delta: ", end)

if timestamp <= datetime.now() <= end:
    st.write("in between")
else:
    st.write("No!")
