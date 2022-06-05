from xmlrpc.client import boolean
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns

from plot_creation import plot_creation_single, plot_creation


def schemepage(email:str, usage_num:int, dimension:int, word_aggregation:str, graph_aggregation:str):
    """
    Create the visualization webpage

    Parameters
    ----------
    email : str
            The email to send the data
    usage_num : str
            The number of time the user used the tool
    dimension : str
            Dimension set by the user
    word_aggregation : str
            Word aggregation set by the user
    graph_aggregation : str
            Graph aggregation set by the user
    """

    scheme = pd.read_csv(f'./meta_features_extracted/{email}_{usage_num}.csv')

    st.write("""We computed our measures using the [ECoL](https://github.com/lpfgarcia/ECoL) (Extended Complexity Library) R package. 
                This implementation characterize the complexity of classification and regression problems based on aspects that quantify the linearity of the data, 
                the presence of informative feature, the sparsity and dimensionality of the datasets.
    """)

    #csv visualization and explaining
    column1, column2 = st.columns([6,1])
    with column1: 
        st.dataframe(scheme)
    with column2:
        st.write('**Dimension**: ', dimension)
        st.write('**Word aggregation**: ', word_aggregation)
        st.write('**Graph aggregation**:', graph_aggregation)

    #Ranking visualization option sidebar
    st.sidebar.subheader("Ranking encoding algorithms performance")
    data_selection_ranking = st.sidebar.multiselect('Select data', ['Encoding time', 'Encoding Memory',	'c1', 'c2', 'cls_coef', 'density', 'f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 'lsc', 'n1', 'n2', 'n3', 't1', 't2', 't3', 't4'], ['Encoding time', 'Encoding Memory',	'c1', 'c2', 'cls_coef', 'density', 'f1', 'f1v', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3', 'lsc', 'n1', 'n2', 'n3', 't1', 't2', 't3', 't4'], key = "ranking_opt")
    st.sidebar.markdown("""---""")

    #ranking dataframe creation
    st.subheader("Ranking encoding algorithms performances")
    source = pd.DataFrame(data={})
    #source['Encoding methods'] = scheme['encoding']

    # check value to add in the dataframe for heatmap

    #encoding time
    if 'Encoding time' in data_selection_ranking:
        source['Time'] = scheme['encoding_time'].rank(method="min", ascending=False)
    #encoding memory
    if 'Encoding Memory' in data_selection_ranking:
        source['memory'] = scheme['encoding_memory'].rank(method="min", ascending=False)
    #c1
    if 'c1' in data_selection_ranking:
        source['C1'] = scheme['c1'].rank(method="min", ascending=False)
    #c2
    if 'c2' in data_selection_ranking:
        source['C2'] = scheme['c2'].rank(method="min", ascending=False)
    #cls_coef
    if 'cls_coef' in data_selection_ranking:
        source['CLS coeff'] = scheme['cls_coef'].rank(method="min", ascending=False)
    #density
    if 'density' in data_selection_ranking:
        source['Density'] = scheme['density'].rank(method="min", ascending=False)
    #f1
    if 'f1' in data_selection_ranking:
        source['F1'] = scheme['f1.mean'].rank(method="min", ascending=False)
    #f1v
    if 'f1v' in data_selection_ranking:
        source['F1v'] = scheme['f1v.mean'].rank(method="min", ascending=False)
    #f2
    if 'f2' in data_selection_ranking:
        source['F2'] = scheme['f2.mean'].rank(method="min", ascending=False)
    #f3
    if 'f3' in data_selection_ranking:
        source['F3'] = scheme['f3.mean'].rank(method="min", ascending=False)
    #f4
    if 'f4' in data_selection_ranking:
        source['F4'] = scheme['f4.mean'].rank(method="min", ascending=False)
    #l1
    if 'l1' in data_selection_ranking:
        source['L1'] = scheme['l1.mean'].rank(method="min", ascending=False)
    #l2
    if 'l2' in data_selection_ranking:
        source['L2'] = scheme['l2.mean'].rank(method="min", ascending=False)
    #l3
    if 'l3' in data_selection_ranking:
        source['L3'] = scheme['l3.mean'].rank(method="min", ascending=False)
    #lsc
    if 'lsc' in data_selection_ranking:
        source['LSC'] = scheme['lsc'].rank(method="min", ascending=False)
    #n1
    if 'n1' in data_selection_ranking:
        source['N1'] = scheme['n1'].rank(method="min", ascending=False)
    #n2
    if 'n2' in data_selection_ranking:
        source['N2'] = scheme['n2.mean'].rank(method="min", ascending=False)
    #n3
    if 'n3' in data_selection_ranking:
        source['N3'] = scheme['n3.mean'].rank(method="min", ascending=False)
    #t1
    if 't1' in data_selection_ranking:
        source['T1'] = scheme['t1.mean'].rank(method="min", ascending=False)
    #t2
    if 't2' in data_selection_ranking:
        source['T2'] = scheme['t2'].rank(method="min", ascending=False)
    #t3
    if 't3' in data_selection_ranking:
        source['T3'] = scheme['t3'].rank(method="min", ascending=False)
    #t4
    if 't4' in data_selection_ranking:
        source['T4'] = scheme['t4'].rank(method="min", ascending=False)
    
    #print heatmap with ranking

    if data_selection_ranking != []:

        #print dataframe
        #st.dataframe(source_t)
        #print heatmap with ranking

        y_axis_labels = scheme['encoding']

        #plt.figure(figsize=(20,2))

        ax = sns.heatmap(source, annot=True, cmap="YlGnBu", yticklabels=y_axis_labels, square=True)
        fig = ax.get_figure()
        #fig.savefig("output.png", dpi=300)
        st.pyplot(fig)
  
    #Data plot visualization option sidebar
    st.sidebar.subheader("Data visualization")
    data = st.sidebar.multiselect('Select data', ['Encoding time', 'Encoding memory', 'Feature vector size'], ['Encoding time'], key="opt")
    option = st.sidebar.selectbox('Select graph',('Bar', 'Point'), key='opt_time')
    st.sidebar.markdown("""---""")

    if data != []:
        st.subheader("Data")
        st.write("""These measurements indicate the computational demands of each algorithm, 
                    as they indicate the time taken and the size occupied.
                    They also indicate the size of the resulting vector, previously chosen.
        """)

    if 'Encoding time' in data:
        # create the plot
        time_plot = plot_creation_single(scheme, option, 'encoding_time', 'Encoding time', False)
        # show plot on streamlit
        st.altair_chart(time_plot, use_container_width=True)

    if 'Encoding memory' in data:
        # create the plot
        memory_plot = plot_creation_single(scheme, option, 'encoding_memory','Encoding memory', False)
        # show plot on streamlit
        st.altair_chart(memory_plot, use_container_width=True)
       
    if 'Feature vector size' in data:
        # create the plot
        vector_size_plot = plot_creation_single(scheme, option, 'feature_vector_size', 'Feature vector size', False)
        # show plot on streamlit
        st.altair_chart(vector_size_plot, use_container_width=True)
        

    # -------------------------------------------------------------------------------------

    #network plot visualization options 
    st.sidebar.subheader("Measures of network")
    data_selection_net = st.sidebar.multiselect("Choose the data", ['Density', 'Clustering coefficient'], ['Density'])
    option_net = st.sidebar.selectbox('Select plot type',('Bar', 'Point'), key = "optnet")
    st.sidebar.markdown("""---""")

    if data_selection_net != []:
        st.subheader("Measures of network")
        st.write("""
            The network measures represent the dataset as a graph and extract structural information from it.
            1. **Density**: Average Density of the network (Density) represents the number of edges in the graph, 
                divided by the maximum number of edges between pairs of data points;
            2. **Clustering Coefficient**: averages the clustering tendency of the vertexes 
                by the ratio of existent edges between its neighbors and the total number of edges that could possibly exist between them;
            """)

    if 'Density' in data_selection_net:
        # create the plot
        density_plot = plot_creation_single(scheme, option_net, 'density', "Average Density of the network", False)
        # show plot on streamlit
        density_chart = st.altair_chart(density_plot, use_container_width=True)

    if 'Clustering coefficient' in data_selection_net:
        # create the plot
        cls_plot = plot_creation_single(scheme, option_net, 'cls_coef', "Clustering coefficient (cls_coef)", False)
        # show plot on streamlit
        cls_chart = st.altair_chart(cls_plot, use_container_width=True)

    if 'Density' in data_selection_net and 'Clustering coefficient' in data_selection_net:
        # delete single plot
        density_chart.empty()
        cls_chart.empty()

        col1, col2 = st.columns(2)

        with col1:
            # show plot on streamlit on column
            density_chart = st.altair_chart(density_plot, use_container_width=True)
        with col2:
            # show plot on streamlit on column
            cls_chart = st.altair_chart(cls_plot, use_container_width=True)
    # -------------------------------------------------------------------------------------

    #c plot visualization options 
    st.sidebar.subheader("Measures of class balance")
    data_selection_c = st.sidebar.multiselect("Choose the data", ['c1', 'c2'], ['c1'])
    option_c = st.sidebar.selectbox('Select plot type',('Bar', 'Point'), key = "optc")
    st.sidebar.markdown("""---""")

    if data_selection_c != []:
        st.subheader("Measures of class balance")
        st.write("""
            These measures capture the differences in the number of examples per class in the dataset. 
            When these differences are severe, problems related to generalization of the ML classification techniques could happen 
            because of the imbalance ratio.
            1. **C1**: The entropy of class proportions (C1) capture the imbalance in a dataset based on the proportions of examples per class;
            2. **C2**: The imbalance ratio (C2) is an index computed for measuring class balance. 
                        This is a version of the measure that is also suited for multiclass classification problems;
        """)

    if 'c1' in data_selection_c:
        # create the plot
        c1_plot = plot_creation_single(scheme, option_c, 'c1', "The entropy of class proportions (C1)", False)
        # show plot on streamlit
        c1_chart = st.altair_chart(c1_plot, use_container_width=True)
    
    if 'c2' in data_selection_c:
        # create the plot
        c2_plot = plot_creation_single(scheme, option_c, 'c2', "The imbalance ratio (C2)", False)
        # show plot on streamlit
        c2_chart = st.altair_chart(c1_plot, use_container_width=True)

    if 'c1' in data_selection_c and 'c2' in data_selection_c:

        # delete single plot
        c1_chart.empty()
        c2_chart.empty()

        col1, col2 = st.columns(2)
        with col1:
            # show plot on streamlit on column
            c1_chart = st.altair_chart(c1_plot, use_container_width=True) 
        with col2:
            # show plot on streamlit on column
            c2_chart = st.altair_chart(c2_plot, use_container_width=True)
    
    # -------------------------------------------------------------------------------------

    #f plot visualization options 
    st.sidebar.subheader("Measures of overlapping")
    data_selection_f = st.sidebar.multiselect("Choose the data", ['f1.mean', 'f1v.mean','f2.mean', 'f3.mean', 'f4.mean'], ['f1.mean'])
    sd_selection_f = st.sidebar.checkbox("Show standard deviation", key = "sdf")
    option_f = st.sidebar.selectbox('Select plot type',('Bar', 'Point'), key = "optf")
    st.sidebar.markdown("""---""")

    if data_selection_f != []:
        st.subheader("Measures of overlapping")
        st.write('''
            The overlapping measures evaluate how informative the available features are to separate the classes. 
            If there is at least one very discriminative feature in the dataset, 
            the problem can be considered simpler than if there is no such an attribute.
            
            1. **F1**: Maximum Fisher’s Discriminant Ratio (F1) measures the overlap between the values of the features 
                and takes the value of the largest discriminant ratio among all the available features.
            2. **F1v**: Directional-vector maximum Fisher’s discriminant ratio (F1v) complements F1 by searching for a vector able to separate two classes after the training examples have been projected into it.
            3. **F2**: Volume of the overlapping region (F2) computes the overlap of the distributions of the fea- tures values within the classes. 
                F2 can be determined by finding, for each feature its minimum and maximum values in the classes.
            4. **F3**: The maximum individual feature efficiency (F3) of each feature is given by the ratio between the number of examples that are not in the overlapping region of two classes and the total number of examples. 
                This measure returns the maximum of the values found among the input features.
            5. **F4**: Collective feature efficiency (F4) get an overview on how various features may work together in data separation. 
                First the most discriminative feature according to F3 is selected and all examples that can be separated by this feature are removed from the dataset. The previous step is repeated on the remaining dataset until all the features have been considered or no example remains. F4 returns the ratio of examples that have been discriminated.

        ''')

    if 'f1.mean' in data_selection_f:
        # create layer
        f1 = alt.layer()
        # create the plot
        f1_plot, f1_plot_sd = plot_creation(scheme, option_f, 'f1', "Fisher's discriminant ratio (F1)")
        # add f1 at the plot
        f1 = alt.layer(f1, f1_plot)

        if sd_selection_f:
            # add sd at the plot
            f1 = alt.layer(f1, f1_plot_sd)
        
        # show plot on streamlit
        f1_chart = st.altair_chart(f1, use_container_width=True)
          
    if 'f1v.mean' in data_selection_f:
        # create layer
        f1v = alt.layer()
        # create the plot
        f1v_plot, f1v_plot_sd = plot_creation(scheme, option_f, 'f1v', "The directional-vector Fisher's discriminant ratio (F1v)")
        # add f1 at the plot
        f1v= alt.layer(f1v, f1v_plot)

        if sd_selection_f:
            # add sd at the plot
            f1v = alt.layer(f1v, f1v_plot_sd)
        
        # show plot on streamlit
        f1v_chart = st.altair_chart(f1v, use_container_width=True)
    
    if 'f2.mean' in data_selection_f:
        # create layer
        f2 = alt.layer()
        # create the plot
        f2_plot, f2_plot_sd = plot_creation(scheme, option_f, 'f2', "Overlapping of the per-class bounding boxes (F2)")
        # add f1 at the plot
        f2 = alt.layer(f2, f2_plot)

        if sd_selection_f:
            # add sd at the plot
            f2 = alt.layer(f2, f2_plot_sd)
        
        # show plot on streamlit
        f2_chart = st.altair_chart(f2, use_container_width=True)

    if 'f1.mean' in data_selection_f and 'f2.mean' in data_selection_f:
        # delete single plot
        f1_chart.empty()
        f2_chart.empty()

        col1, col2 = st.columns(2)
        with col1:
            # show plot on streamlit on column
           f1_chart = st.altair_chart(f1, use_container_width=True)
        with col2:
            # show plot on streamlit on column
            f2_chart = st.altair_chart(f2, use_container_width=True)
        
    if 'f3.mean' in data_selection_f:
        # create layer
        f3 = alt.layer()
        # create the plot
        f3_plot, f3_plot_sd = plot_creation(scheme, option_f, 'f3', "Maximum individual feature efficiency (F3)")
        # add f1 at the plot
        f3 = alt.layer(f3, f3_plot)

        if sd_selection_f:
            # add sd at the plot
            f3 = alt.layer(f3, f3_plot_sd)
        
        # show plot on streamlit
        f3_chart = st.altair_chart(f3, use_container_width=True)

    if 'f4.mean' in data_selection_f:
        # create layer
        f4 = alt.layer()
        # create the plot
        f4_plot, f4_plot_sd = plot_creation(scheme, option_f, 'f4', "Collective feature efficiency (F4)")
        # add f1 at the plot
        f4 = alt.layer(f4, f4_plot)

        if sd_selection_f:
            # add sd at the plot
            f4 = alt.layer(f4, f4_plot_sd)
        
        # show plot on streamlit
        f4_chart = st.altair_chart(f4, use_container_width=True)

    if 'f3.mean' in data_selection_f and 'f4.mean' in data_selection_f:
         # delete single plot
        f3_chart.empty()
        f4_chart.empty()

        col1, col2 = st.columns(2)
        with col1:
            f3_chart = st.altair_chart(f3, use_container_width=True)
        with col2:
            f4_chart = st.altair_chart(f4, use_container_width=True)

    # -------------------------------------------------------------------------------------

    #l plot visualization options 
    st.sidebar.subheader("Measures of linearity")
    data_selection_l = st.sidebar.multiselect("Choose the data", ['l1.mean', 'l2.mean', 'l3.mean'], ['l1.mean'])
    sd_selection_l = st.sidebar.checkbox("Show standard deviation", key = "sdl")
    option_l = st.sidebar.selectbox('Select plot type',('Bar', 'Point'), key = "optl")
    st.sidebar.markdown("""---""")
    
    if data_selection_l != []:
        st.subheader("Measures of linearity")
        st.write('''
            The linearity measures try to quantify if it is possible to separate the labels by a hyperplane or linear function. 
            The underlying assumption is that a linearly separable problem can be considered simpler than a problem requiring a non-linear decision boundary.
        ''')

    if 'l1.mean' in data_selection_l:
        # create layer
        l1 = alt.layer()
        # create the plot
        l1_plot, l1_plot_sd = plot_creation(scheme, option_l, 'l1', "Sum of the error distance by linear programming (L1)")
        # add l1 at the plot
        l1 = alt.layer(l1, l1_plot)

        if sd_selection_l:
            # add sd at the plot
            l1 = alt.layer(l1, l1_plot_sd)
    
        # show plot on streamlit
        l1_chart = st.altair_chart(l1, use_container_width=True)
    
    if 'l2.mean' in data_selection_l: 
        # create layer
        l2 = alt.layer()
        # create the plot
        l2_plot, l2_plot_sd = plot_creation(scheme, option_l, 'l2', "Error rate of linear classifier (L2)")
        # add l1 at the plot
        l2 = alt.layer(l2, l2_plot)

        if sd_selection_l:
            # add sd at the plot
            l2 = alt.layer(l2, l2_plot_sd)
        
        # show plot on streamlit
        l2_chart = st.altair_chart(l2, use_container_width=True)

    if 'l3.mean' in data_selection_l:
        # create layer
        l3 = alt.layer()
        # create the plot
        l3_plot, l3_plot_sd = plot_creation(scheme, option_l, 'l3', "Nonlinearity of a linear classifier (L3)")
        # add l1 at the plot
        l3 = alt.layer(l3, l3_plot)

        if sd_selection_l:
            # add sd at the plot
            l3 = alt.layer(l3, l3_plot_sd)
        
        # show plot on streamlit
        l3_chart = st.altair_chart(l3, use_container_width=True)  

    # -------------------------------------------------------------------------------------

    #n plot visualization options 
    st.sidebar.subheader("Measures of neighborhood")
    data_selection_n = st.sidebar.multiselect("Choose the data", ['n1', 'n2.mean', 'n3.mean', 'lsc'], ['n1'])
    sd_selection_n = st.sidebar.checkbox("Show standard deviation", key = "sdn")
    option_n = st.sidebar.selectbox('Select plot type',('Bar', 'Point'), key = "optn")
    st.sidebar.markdown("""---""")

    if data_selection_n != []:
        st.subheader("Measures of neighborhood")
        st.write('''
            The Neighborhood measures analyze the neighborhoods of the data items and try to capture class overlapping and the shape of the decision boundary. 
            They work over a distance matrix storing the distances between all pairs of data points in the dataset.

            1. **N1**: Fraction of borderline points (N1) computes the percentage of vertexes incident to edges connecting examples of opposite classes in a Minimum Spanning Tree (MST).
            2. **N2**: Ratio of intra/extra class nearest neighbor distance (N2) computes the ratio of two sums: intra-class and inter-class. 
                The former corresponds to the sum of the distances between each example and its closest neighbor from the same class. 
                The later is the sum of the distances between each example and its closest neighbor from another class (nearest enemy).
            3. **N3**: Error rate of the nearest neighbor (N3) classifier corresponds to the error rate of a one Nearest Neighbor (1NN) classifier, 
                estimated using a leave-one-out procedure in dataset.
            4. **LSC**: Local Set Average Cardinality (LSC) is based on Local Set (LS) and defined as the set of points 
                from the dataset whose distance of each example is smaller than the distance from the exemples of the different class. 
                LSC is the average of the LS.
        ''')

    if 'n1' in data_selection_n:
        # create the plot
        n1_plot = plot_creation_single(scheme, option_n, 'n1', "Fraction of borderline points (N1)", False)
        # show plot on streamlit
        n1_chart = st.altair_chart(n1_plot, use_container_width=True)
    
    if 'n2.mean' in data_selection_n:
        # create layer
        n2 = alt.layer()
        # create the plot
        n2_plot, n2_plot_sd = plot_creation(scheme, option_l, 'n2', "Ratio of intra/extra class nearest neighbor distance (N2)")
        # add l1 at the plot
        n2 = alt.layer(n2, n2_plot)

        if sd_selection_n:
            # add sd at the plot
            n2 = alt.layer(n2, n2_plot_sd)
        
        # show plot on streamlit
        n2_chart = st.altair_chart(n2, use_container_width=True)

    if 'n3.mean' in data_selection_n:
        # create layer
        n3 = alt.layer()
        # create the plot
        n3_plot, n3_plot_sd = plot_creation(scheme, option_l, 'n3', "Error rate of the nearest neighbor (N3)")
        # add n3 at the plot
        n3 = alt.layer(n3, n3_plot)

        if sd_selection_n:
            # add sd at the plot
            n3 = alt.layer(n3, n3_plot_sd)
        
        # show plot on streamlit
        n3_chart = st.altair_chart(n3, use_container_width=True)

    if 'lsc' in data_selection_n:
        # create the plot
        lsc_plot = plot_creation_single(scheme, option_l, 'lsc', "Local Set Average Cardinality (LSC)", False)
        # show plot on streamlit
        lsc_chart = st.altair_chart(lsc_plot, use_container_width=True)
    
    # -------------------------------------------------------------------------------------

    #t plot visualization options 
    st.sidebar.subheader("Measures of dimensionality")
    data_selection_t = st.sidebar.multiselect("Choose the data", ['t1.mean', 't2', 't3', 't4'], ['t1.mean'])
    sd_selection_t = st.sidebar.checkbox("Show standard deviation", key = "sdt")
    option_t = st.sidebar.selectbox('Select plot type',('Bar', 'Point'), key = "optt")
    st.sidebar.markdown("""---""")

    if data_selection_t != []:
        st.subheader("Measures of dimensionality")
        st.write('''
            These measures give an indicative of data sparsity. 
            They capture how sparse a datasets tend to have regions of low density. 
            These regions are know to be more difficult to extract good classification and regression models.
            1. **T2**: Average number of points per dimension (T2) is given by the ratio between the number of examples and dimensionality of the dataset.
            2. **T3**: Average number of points per PCA (T3) is similar to T2, 
                but uses the number of PCA com- ponents needed to represent 95 variability as the base of data sparsity assessment.
            3. **T4**: Ratio of the PCA Dimension to the Original (T4) estimates the proportion of relevant and the original dimensions for a dataset.
        ''')

    if 't1.mean' in data_selection_t:
       # create layer
        t1 = alt.layer()
        # create the plot
        t1_plot, t1_plot_sd = plot_creation(scheme, option_t, 't1', "Fraction of hyperspheres covering data (T1)")
        # add n3 at the plot
        t1 = alt.layer(t1, t1_plot)

        if sd_selection_t:
            # add sd at the plot
            t1 = alt.layer(t1, t1_plot_sd)
        
        # show plot on streamlit
        t1_chart = st.altair_chart(t1, use_container_width=True)

    if 't2' in data_selection_t:
        # create the plot
        t2_plot = plot_creation_single(scheme, option_t, 't2', "Average number of points per dimension (T2)", False)
        # show plot on streamlit
        t2_chart = st.altair_chart(t2_plot, use_container_width=True)

    if 't1.mean' in data_selection_t and 't2' in data_selection_t:
      
        # delete single barplot
        t1_chart.empty()
        t2_chart.empty()

        col1, col2 = st.columns(2)
        with col1:
            # show plot on streamlit on left column
            t1_chart = st.altair_chart(t1, use_container_width=True)
            # show plot on streamlit on right column
        with col2:
            t2_chart = st.altair_chart(t2_plot, use_container_width=True)
            
    if 't3' in data_selection_t:
        # create the plot
        t3_plot = plot_creation_single(scheme, option_t, 't3', "Average number of points per PCA (T3)", False)
        # show plot on streamlit
        t3_chart = st.altair_chart(t3_plot, use_container_width=True)

    if 't4' in data_selection_t:
        # create the plot
        t4_plot = plot_creation_single(scheme, option_t, 't4', "Ratio of the PCA Dimension to the Original (T4)", False)
        # show plot on streamlit
        t4_chart = st.altair_chart(t4_plot, use_container_width=True)
    
    if 't3' in data_selection_t and 't4' in data_selection_t:
        
        # delete single barplot
        t3_chart.empty()
        t4_chart.empty()

        col1, col2 = st.columns(2)
        with col1:
            # show plot on streamlit on column
            t3_chart = st.altair_chart(t3_plot, use_container_width=True)
        with col2:
            # show plot on streamlit on column
            t4_chart = st.altair_chart(t4_plot, use_container_width=True)
