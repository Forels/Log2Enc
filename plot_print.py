import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from zipfile import ZipFile
import os


def plot_print(email:str, usage_num:int):
    """
    Create the altair plot, save to .pdf file and create a .zip file 

    Parameters
    ----------
    email : str
        The email of the user
    usage_num : int
        The number of time that the user uses the tool
    """

    scheme = pd.read_csv(f'./meta_features_extracted/{email}_{usage_num}.csv')

    sns.set_style("whitegrid")

    # Heatmap 

    source = pd.DataFrame(data={})

    #encoding time
    source['Time'] = scheme['encoding_time'].rank(method="min", ascending=False)
    #encoding memory
    source['memory'] = scheme['encoding_memory'].rank(method="min", ascending=False)
    #c1
    source['C1'] = scheme['c1'].rank(method="min", ascending=False)
    #c2
    source['C2'] = scheme['c2'].rank(method="min", ascending=False)
    #cls_coef
    source['CLS coeff'] = scheme['cls_coef'].rank(method="min", ascending=False)
    #density
    source['Density'] = scheme['density'].rank(method="min", ascending=False)
    #f1
    source['F1'] = scheme['f1.mean'].rank(method="min", ascending=False)
    #f1v
    source['F1v'] = scheme['f1v.mean'].rank(method="min", ascending=False)
    #f2
    source['F2'] = scheme['f2.mean'].rank(method="min", ascending=False)
    #f3
    source['F3'] = scheme['f3.mean'].rank(method="min", ascending=False)
    #f4
    source['F4'] = scheme['f4.mean'].rank(method="min", ascending=False)
    #l1
    source['L1'] = scheme['l1.mean'].rank(method="min", ascending=False)
    #l2
    source['L2'] = scheme['l2.mean'].rank(method="min", ascending=False)
    #l3
    source['L3'] = scheme['l3.mean'].rank(method="min", ascending=False)
    #lsc
    source['LSC'] = scheme['lsc'].rank(method="min", ascending=False)
    #n1
    source['N1'] = scheme['n1'].rank(method="min", ascending=False)
    #n2
    source['N2'] = scheme['n2.mean'].rank(method="min", ascending=False)
    #n3
    source['N3'] = scheme['n3.mean'].rank(method="min", ascending=False)
    #t1
    source['T1'] = scheme['t1.mean'].rank(method="min", ascending=False)
    #t2
    source['T2'] = scheme['t2'].rank(method="min", ascending=False)
    #t3
    source['T3'] = scheme['t3'].rank(method="min", ascending=False)
    #t4
    source['T4'] = scheme['t4'].rank(method="min", ascending=False)

    y_axis_labels = scheme['encoding']

    fig_hm, axes = plt.subplots(1, sharey=True, figsize=(15,10))
    fig_hm.suptitle('Heatmap', weight= "bold", size='xx-large')

    a = sns.heatmap(data=source, annot=True, cmap="YlGnBu", yticklabels=y_axis_labels, square=True)

    plt.close()
    # -------------------------------------------------------------------------------------------

    # Time - Memory - Feature vector size
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15,10))
    fig.suptitle('Data', weight= "bold", size='xx-large')

    a = sns.barplot(ax=axes[0], x="encoding_time", y="encoding", data=scheme, palette="Blues_d")
    a.set_xlabel('Encoding time', fontsize='large')
    a.set_ylabel('Encoding methods',  fontsize='large')

    b = sns.barplot(ax=axes[1], x="encoding_memory", y="encoding", data=scheme, palette="Blues_d")
    b.set_xlabel('Encoding memory', fontsize='large')
    b.set_ylabel('Encoding methods',  fontsize='large')

    c = sns.barplot(ax=axes[2], x="feature_vector_size", y="encoding", data=scheme, palette="Blues_d")
    c.set_xlabel('Feature vector size', fontsize='large')
    c.set_ylabel('Encoding methods',  fontsize='large')

    plt.close()
    # -------------------------------------------------------------------------------------------

    # Density - cls 
    fig_net, axes = plt.subplots(1, 2, sharey=True, figsize=(15,10))
    fig_net.suptitle('Measure of network', weight= "bold", size='xx-large')

    a = sns.barplot(ax=axes[0], x="density", y="encoding", data=scheme, palette="Blues_d")
    a.set_xlabel('Average Density of the network', fontsize='large')
    a.set_ylabel('Encoding methods',  fontsize='large')

    b = sns.barplot(ax=axes[1], x="cls_coef", y="encoding", data=scheme, palette="Blues_d")
    b.set_xlabel('Clustering coefficient (cls_coef)', fontsize='large')
    b.set_ylabel('Encoding methods',  fontsize='large')

    plt.close()
    # -------------------------------------------------------------------------------------------

    # C1 - C2
    fig_c, axes = plt.subplots(1, 2, sharey=True, figsize=(15,10))
    fig_c.suptitle('Measures of class balance', weight= "bold", size='xx-large')

    a = sns.barplot(ax=axes[0], x="c1", y="encoding", data=scheme, palette="Blues_d")
    a.set_xlabel('The entropy of class proportions (C1)', fontsize='large')
    a.set_ylabel('Encoding methods',  fontsize='large')

    b = sns.barplot(ax=axes[1], x="c2", y="encoding", data=scheme, palette="Blues_d")
    b.set_xlabel('The imbalance ratio (C2)', fontsize='large')
    b.set_ylabel('Encoding methods',  fontsize='large')

    plt.close()
    # -------------------------------------------------------------------------------------------

    # F1 - F2 - F3 - F4
    fig_f, axes = plt.subplots(2, 2, sharey=True, figsize=(15,10))
    fig_f.suptitle('Measures of overlapping', weight= "bold", size='xx-large')

    a = sns.barplot(ax=axes[0, 0], x="f1.mean", y="encoding", data=scheme, palette="Blues_d")
    a.set_xlabel("Fisher's discriminant ratio (F1)", fontsize='large')
    a.set_ylabel('Encoding methods',  fontsize='large')

    b = sns.barplot(ax=axes[0, 1], x="f2.mean", y="encoding", data=scheme, palette="Blues_d")
    b.set_xlabel("Overlapping of the per-class bounding boxes (F2)", fontsize='large')
    b.set_ylabel('Encoding methods',  fontsize='large')

    c = sns.barplot(ax=axes[1, 0], x="f3.mean", y="encoding", data=scheme, palette="Blues_d")
    c.set_xlabel("Maximum individual feature efficiency (F3)", fontsize='large')
    c.set_ylabel('Encoding methods',  fontsize='large')

    d = sns.barplot(ax=axes[1, 1], x="f4.mean", y="encoding", data=scheme, palette="Blues_d")
    d.set_xlabel("Collective feature efficiency (F4)", fontsize='large')
    d.set_ylabel('Encoding methods',  fontsize='large')

    plt.close()
    # -------------------------------------------------------------------------------------------

    # L1.mean - L2.mean - L3.mean - LSC
    fig_l, axes = plt.subplots(1, 3, sharey=True, figsize=(15,10))
    fig_l.suptitle('Measures of linearity', weight= "bold", size='xx-large')

    a = sns.barplot(ax=axes[0], x="l1.mean", y="encoding", data=scheme, palette="Blues_d")
    a.set_xlabel('Sum of the error distance by linear programming (L1)', fontsize='large')
    a.set_ylabel('Encoding methods',  fontsize='large')

    b = sns.barplot(ax=axes[1], x="l2.mean", y="encoding", data=scheme, palette="Blues_d")
    b.set_xlabel('Error rate of linear classifier (L2)', fontsize='large')
    b.set_ylabel('Encoding methods',  fontsize='large')

    c = sns.barplot(ax=axes[2], x="l3.mean", y="encoding", data=scheme, palette="Blues_d")
    c.set_xlabel('Nonlinearity of a linear classifier (L3)', fontsize='large')
    c.set_ylabel('Encoding methods',  fontsize='large')

    plt.close()
    # -------------------------------------------------------------------------------------------

    # N1 - N2.mean - N3.mean - LSC
    fig_n, axes = plt.subplots(2, 2, sharey=True, figsize=(15,10))
    fig_n.suptitle('Measures of neighborhood', weight= "bold", size='xx-large')

    a = sns.barplot(ax=axes[0, 0], x="n1", y="encoding", data=scheme, palette="Blues_d")
    a.set_xlabel('Fraction of borderline points (N1)', fontsize='large')
    a.set_ylabel('Encoding methods',  fontsize='large')

    b = sns.barplot(ax=axes[0, 1], x="n2.mean", y="encoding", data=scheme, palette="Blues_d")
    b.set_xlabel('Ratio of intra/extra class nearest neighbor distance (N2)', fontsize='large')
    b.set_ylabel('Encoding methods',  fontsize='large')

    c = sns.barplot(ax=axes[1, 0], x="n3.mean", y="encoding", data=scheme, palette="Blues_d")
    c.set_xlabel('Error rate of the nearest neighbor (N3)', fontsize='large')
    c.set_ylabel('Encoding methods',  fontsize='large')

    c = sns.barplot(ax=axes[1, 1], x="lsc", y="encoding", data=scheme, palette="Blues_d")
    c.set_xlabel('Local Set Average Cardinality (LSC)', fontsize='large')
    c.set_ylabel('Encoding methods',  fontsize='large')
    
    plt.close()
    # -------------------------------------------------------------------------------------------

    # T1.mean - T2 - T3 - T4 
    fig_t, axes = plt.subplots(2, 2, sharey=True, figsize=(15,10))
    fig_t.suptitle('Measures of dimensionality', weight= "bold", size='xx-large')

    a = sns.barplot(ax=axes[0, 0], x="t1.mean", y="encoding", data=scheme, palette="Blues_d")
    a.set_xlabel('Fraction of hyperspheres covering data (T1)', fontsize='large')
    a.set_ylabel('Encoding methods',  fontsize='large')

    b = sns.barplot(ax=axes[0, 1], x="t2", y="encoding", data=scheme, palette="Blues_d")
    b.set_xlabel('Average number of points per dimension (T2)', fontsize='large')
    b.set_ylabel('Encoding methods',  fontsize='large')

    c = sns.barplot(ax=axes[1, 0], x="t3", y="encoding", data=scheme, palette="Blues_d")
    c.set_xlabel('Average number of points per PCA (T3)', fontsize='large')
    c.set_ylabel('Encoding methods',  fontsize='large')

    d = sns.barplot(ax=axes[1, 1], x="t4", y="encoding", data=scheme, palette="Blues_d")
    d.set_xlabel('Ratio of the PCA Dimension to the Original (T4)', fontsize='large')
    d.set_ylabel('Encoding methods',  fontsize='large')

    plt.close()
    # -------------------------------------------------------------------------------------------

    pp = PdfPages(f"{email}_{usage_num}.pdf")

    pp.savefig(fig_hm, orientation='landscape', bbox_inches='tight', pad_inches=1)
    pp.savefig(fig, orientation='landscape', bbox_inches='tight', pad_inches=1)
    pp.savefig(fig_net, orientation='landscape', bbox_inches='tight', pad_inches=1)
    pp.savefig(fig_c, orientation='portrait', bbox_inches='tight', pad_inches=1)
    pp.savefig(fig_f, orientation='portrait', bbox_inches='tight', pad_inches=1)
    pp.savefig(fig_l, orientation='portrait', bbox_inches='tight', pad_inches=1)
    pp.savefig(fig_n, orientation='portrait', bbox_inches='tight', pad_inches=1)
    pp.savefig(fig_t, orientation='portrait', bbox_inches='tight', pad_inches=1)
    pp.close()

    # Create a ZipFile Object
    zipObj = ZipFile(f'{email}_{usage_num}.zip', 'w')

    # Add multiple files to the zip
    zipObj.write(f'./meta_features_extracted/{email}_{usage_num}.csv')
    zipObj.write(f'{email}_{usage_num}.pdf')
 
    # close the Zip File
    zipObj.close()
