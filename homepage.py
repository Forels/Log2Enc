from sqlite3 import Timestamp
from turtle import width

from requests import session
import streamlit as st
from schemepage import schemepage
from PIL import Image
import gspread
import os
import fnmatch
import numpy as np
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util import constants
from datetime import *

from feature_extract import feature_extract, feature_save
from preprocessing import process_model
from preprocessing import graph_embeddings
from preprocessing import word_embeddings

from compute_encoding import alignment, tokenreplay, log_skeleton, \
countvectorizer, doc2vec, hashvectorizer, onehot, tfidfvectorizer, word2vec_cbow, word2vec_skipgram, \
boostne, deepwalk, diff2vec, glee, graphwave, grarep, hope, laplacianeigenmaps, netmf, nmfadmm, node2vec_, nodesketch, role2vec, walklets
    #glove_

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from os.path import basename

from plot_print import plot_print

# Function to save uploaded event log
def save_uploadedfile(uploadedfile:object, path:str):
    """
    Save the event log uploaded by the user

    Parameters
    ----------
    uploadedfile : str
            The event log uploaded by the user
    path : str
            The path of the event log uploaded
    """

    if not os.path.exists("event_logs_uploaded"):
        os.makedirs("event_logs_uploaded")

    with open(os.path.join("event_logs_uploaded",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
        print("file salvato")

        st.write(uploadedfile.name)    

        # import log
        log = pm4py.read_xes(f'{path}/{uploadedfile.name}')
        params = {
            constants.PARAMETER_CONSTANT_CASEID_KEY: 'case:concept:name',
            constants.PARAMETER_CONSTANT_ACTIVITY_KEY: 'concept:name',
            constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: 'timestamp'}
        df = log_converter.apply(log, parameters=params, variant=log_converter.Variants.TO_DATA_FRAME) 

        # check if there is a label, if not create them and set to 'normal'
        if 'label' not in df:
            df['label'] = np.nan    
            df['label'] = df['label'].replace(np.nan, "normal")

        log = pm4py.convert_to_event_log(df)

        xes_exporter.apply(log, f'{path}/{uploadedfile.name}')
                
        os.remove(f'{path}/{uploadedfile.name}')

    return df, log

def homepage():
    """
    Create the homepage
    """
    
    # Config variable
    path = './event_logs_uploaded'
    # Variable to check email
    start_elaboration = False
    pattern = "*@*"
    # Connecting to google sheet
    gc = gspread.service_account(filename='creds.json')
    sh = gc.open('database').sheet1

    if 'alpha' not in st.session_state:
        st.session_state['alpha'] = False

    # Homepage container
    main = st.empty()
    with main.container():

        # Title of the homepage
        st.title("Log2Enc")

        # Log2Enc logo impagination
        c1, c2, c3 = st.columns([1,4,1])
        image = Image.open('./images/l2e.png')
        c2.image(image)

        # description
        description = st.write('''Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
                    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
                    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
                    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.''')

        #file uploader 
        datafile = st.file_uploader('Choose an event log', type = 'xes')


    # when the user upload en event log, a sidebar appear
    if datafile is not None or st.session_state['alpha'] is True:
        # sidebar container
        placeholder = st.sidebar.empty()
    
        with placeholder.container():
        
            # dimension
            dimension = st.select_slider('Select the dimension', options=[2, 4, 8, 16, 32, 64, 128, 256]) 
            st.write('The selected dimension is', dimension) 

            # word aggregation drop menu 
            word_aggregation = st.selectbox('Select the aggregation for word embedding', ('Average', 'Max'))

            # graph aggreagation drop menu
            graph_aggregation = st.selectbox('Select the aggregation for graph embedding', ('Edge', 'Node'))
                
            # Node aggregation - node/average or node/max
            if graph_aggregation == "Node":
                graph_aggregation = st.selectbox('Select the aggregation for graph embedding', ('Node/Average', 'Node/Max'))
                
            # Edge aggregation - edge/average/average, edge/average/max, edge/weightedl1/average, edge/weightedl1/max, edge/weightedl2/average, edge/weightedl2/max, edge/hadamard/average, edge/hadamard/max
            if graph_aggregation == "Edge":
                graph_aggregation = st.selectbox('Select the aggregation for graph embedding', ('Edge/Average/Average', 'Edge/Average/Max', 'Edge/Weightedl1/Average', 'Edge/Weightedl1/Max', 'Edge/Weightedl2/Average', 'Edge/Weightedl2/Max', 'Edge/Hadamard/Average', 'Edge/Hadamard/Max', ))

            # email text input
            email = st.text_input('Submit your email address to upload an event log', 'example@mail.com')
        
            #confirm button
            confirm_button = st.button('Confirm', key='conf')

        if confirm_button:
            # Check if mail == *@*
            if fnmatch.fnmatch(email, pattern):
                
                # find email in the google sheet
                email_cell = sh.find(email)
                            
                # email already submitted
                if email_cell is not None:
                    
                    usage_num = int(sh.cell(email_cell.row,2).value)
                    timestamp_start = datetime.fromisoformat(str(sh.cell(email_cell.row,3).value))
                    timestamp_end = timestamp_start + timedelta(minutes=5)                        

                    # if timestamp is in this time frame, the tool can be used 3 times
                    if timestamp_start <= datetime.now() <= timestamp_end:

                        # check if email is used more than 3 times
                        if usage_num > 2:
                            st.error(f"Your email has already been used three times in the last 5 minutes. You have to wait until {timestamp_end.strftime('%H:%M')}")
                        
                        # email is used less than 3 times
                        else:
                            usage_num = usage_num + 1
                            sh.update_cell(email_cell.row,2, usage_num)  
                            start_elaboration = True

                    # if timestamp isn't in this time frame, reset of the data 
                    else:
                        usage_num = 1
                        timestamp = datetime.now()
                        sh.update_cell(email_cell.row,2, usage_num)
                        sh.update_cell(email_cell.row,3, timestamp.isoformat())
                        start_elaboration = True

                # email used for the first time
                else:
                    usage_num = 1
                    timestamp = datetime.now().isoformat()
                    sh.append_row([email,usage_num,timestamp])
                    
                    start_elaboration = True

            # all check are passed, execute the elaboration of the data
            if start_elaboration is True:        
                
                with st.spinner('Wait for it...'):
                    # save event log 

                    if datafile is not None:
                        df, log = save_uploadedfile(datafile, path)
                    # PREPROCESSING PHASE ..............................................................

                    # PROCESS MODEL
                    ids_pm, traces_pm, y_pm, net, initial_marking, final_marking = process_model(df, log)
                    print("------------ PROCESS MODEL ------------")                    

                    # WORD EMBEDDINGS    
                    ids_we, traces_we, y_we = word_embeddings(df)
                    print("------------ WORD EMBEDDINGS ------------")

                    # GRAPH EMBEDDIGNS
                    graph, ids_ge, traces_ge, y_ge = graph_embeddings(datafile, df, log)
                    print("------------ GRAPH EMBEDDINGS ------------")
                    
                    # PROCESS MODEL .....................................................    
                    
                    metrics = []
                        
                    out_df, file_name, encoding = alignment(log, datafile, ids_pm, traces_pm, y_pm, net, initial_marking, final_marking)
                    print("Alignment ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = tokenreplay(log, datafile, ids_pm, traces_pm, y_pm, net, initial_marking, final_marking)
                    print("Token Replay ------------")
                    metrics, ft =  feature_extract(out_df, file_name, encoding, metrics)

                    out_df, file_name, encoding = log_skeleton(log, datafile, ids_pm, traces_pm, y_pm, net, initial_marking, final_marking)
                    print("Log Skeleton ------------")
                    metrics, ft =  feature_extract(out_df, file_name, encoding, metrics)

                    # WORD EMBEDDINGS .....................................................    
                        
                    out_df, file_name, encoding = countvectorizer(datafile, ids_we, traces_we, y_we)
                    print("Count2vec ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = doc2vec(datafile, df, dimension)
                    print("Doc2vec ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = hashvectorizer(datafile, ids_we, traces_we, y_we, dimension)
                    print("Hashvectorizer ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    #print("Glove ------------")
                    #out_df, file_name, encoding = glove_(datafile, df, dimension, word_aggregation)
                    #metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = onehot(datafile, ids_we, traces_we, y_we)
                    print("One Hot ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = tfidfvectorizer(datafile, ids_we, traces_we, y_we)
                    print("TFIDF Vectorizer ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = word2vec_cbow(datafile, df, dimension, word_aggregation)
                    print("Word2vec cbow ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                            
                    out_df, file_name, encoding = word2vec_skipgram(datafile, df, dimension, word_aggregation)
                    print("Word2vec skipgram ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)

                    # GRAPH EMBEDDIGNS ...................................................

                    out_df, file_name, encoding = boostne(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)
                    print("BoostNE ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)    

                    out_df, file_name, encoding = deepwalk(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)
                    print("Deep Walk ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                    
                    out_df, file_name, encoding = diff2vec(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)                
                    print("Diff2vec ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                            
                    out_df, file_name, encoding = glee(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)   
                    print("Glee ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)

                    out_df, file_name, encoding = graphwave(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)   
                    print("Graph Wave  ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)

                    out_df, file_name, encoding = grarep(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)  
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                    print("Grarep  ------------")
                        
                    out_df, file_name, encoding = hope(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)
                    print("Hope  ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                            
                    out_df, file_name, encoding = laplacianeigenmaps(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)   
                    print("Laplacian eigen maps ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = netmf(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)  
                    print("NetMF ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = nmfadmm(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation) 
                    print("Nmfadmm ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = node2vec_(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)   
                    print("Node2vec_ ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = nodesketch(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)   
                    print("Node Sketch ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = walklets(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)
                    print("Walklets ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)
                        
                    out_df, file_name, encoding = role2vec(datafile, graph, ids_ge, traces_ge, y_ge, dimension, graph_aggregation)
                    print("Role2vec ------------")
                    metrics, ft = feature_extract(out_df, file_name, encoding, metrics)

                    # extract meta feature and create the .csv file
                    feature_save(metrics, email, usage_num, ft, traces_we)
                    
                    # create the plot, save in .pdf and create the .zip file to send
                    plot_print(email, usage_num)

                    #sending email
                    #Establish SMTP Connection
                    s = smtplib.SMTP('smtp.gmail.com', 587) 
                    #Start TLS based SMTP Session
                    s.starttls() 
                    #Login Using Your Email ID & Password
                    s.login("f.gionata@gmail.com", "itkvkzeuhafklgci")
                    #To Create Email Message in Proper Format
                    msg = MIMEMultipart()

                    #Setting Email Parameters
                    msg['From'] = "f.gionata@gmail.com"     
                    msg['Subject'] = "Notification"
                        
                    #Email Body Content
                    message = """
                    Hi, \n
                    the data processing is completed. \n
                    You can find attached the resulting files, or consult the web page for more details
                    """
                        
                    #Add Message To Email Body
                    msg.attach(MIMEText(message, 'html'))
                        
                    #To Attach File
                    fileName = f'{email}_{usage_num}.zip'
                    file = open(fileName, "rb")
                    fileBaseName = basename(fileName)
                    part = MIMEApplication(file.read(), Name = fileBaseName)
                    part.add_header('Content-Disposition', 'attachment; filename="' + fileBaseName + '"')
                    msg.attach(part)

                    msg['To'] = email   
                    #To Send the Email
                    s.send_message(msg)
                    
                    #Terminating the SMTP Session
                    s.quit()  

                    # counter to stay in the schemepage
                    st.session_state['a_counter'] += 1
                    # empty the homepage
                    main.empty()
                    placeholder.empty()

                    # update the session_state values
                    st.session_state['email'] = email
                    st.session_state['usage_num'] = usage_num
                    st.session_state['dimension'] = dimension
                    st.session_state['word_aggregation'] = word_aggregation
                    st.session_state['graph_aggregation'] = graph_aggregation

                # run the schemepage - visualization
                schemepage(email, usage_num, dimension, word_aggregation, graph_aggregation)
                
                # remove the .zip and .pdf file
                os.remove(f'{email}_{usage_num}.pdf')
                os.remove(f'{email}_{usage_num}.zip')
               
