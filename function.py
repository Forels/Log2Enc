from turtle import color
from xmlrpc.client import boolean
import altair as alt
import streamlit as st
import pandas as pd

def plot_creation(scheme:object, option:str, value:str):
    """
    Parameters
    ----------
    scheme : object
            The dataframe from .csv file
    option : str
            The type of the plot choose by the user
    value : str
            The value to use in the plot

    Returns
    -------
    chart_mean, chart_sd
        the chart of the mean value and the sd value
    """

    # if the value are between 0 and 1 the plot are not interactive to limit the domain value of the x axes
    if all(scheme[f'{value}.mean']) <= 1.0:

        if option == 'Point':
        #value.mean
            chart_mean = alt.Chart(scheme).mark_point(size=100).encode(
                        x=alt.X(f'{value}'r"\.mean:Q", axis=alt.Axis(title=f'{value}'.upper())), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.mean:Q"],
                        color=alt.Color(f'{value}'r"\.mean:Q", scale=alt.Scale(scheme='blues'), legend=None)
            )
            #value.sd
            chart_sd = alt.Chart(scheme).mark_line().encode(
                        x=alt.X(f'{value}'r"\.mean:Q"), 
                        x2=alt.X2(f'{value}'r"\.sd:Q"), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.sd:Q"],
            )

        if option == 'Bar':
            #value.mean
            chart_mean = alt.Chart(scheme).mark_bar().encode(
                        x=alt.X(f'{value}'r"\.mean:Q", axis=alt.Axis(title=f'{value}'.upper())), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.mean:Q"],
                        color=alt.Color(f'{value}'r"\.mean:Q", scale=alt.Scale(scheme='blues'), legend=None)
            )
            #value.sd
            chart_sd = alt.Chart(scheme).mark_rule().encode(
                        x=alt.X(f'{value}'r"\.mean:Q"), 
                        x2=alt.X2(f'{value}'r"\.sd:Q"), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.sd:Q"]
            )

        return chart_mean, chart_sd

    if all(scheme[f'{value}.mean']) >= 1.0:
    
        if option == 'Point':
            #value.mean
            chart_mean = alt.Chart(scheme).mark_point(size=100).encode(
                        x=alt.X(f'{value}'r"\.mean:Q", axis=alt.Axis(title=f'{value}'.upper())), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.mean:Q"],
                        color=alt.Color(f'{value}'r"\.mean:Q", scale=alt.Scale(scheme='blues'), legend=None)
            ).interactive()
            #value.sd
            chart_sd = alt.Chart(scheme).mark_line().encode(
                        x=alt.X(f'{value}'r"\.mean:Q"), 
                        x2=alt.X2(f'{value}'r"\.sd:Q"), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.sd:Q"],
            ).interactive()

        if option == 'Bar':
            #value.mean
            chart_mean = alt.Chart(scheme).mark_bar().encode(
                        x=alt.X(f'{value}'r"\.mean:Q", axis=alt.Axis(title=f'{value}'.upper())), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.mean:Q"],
                        color=alt.Color(f'{value}'r"\.mean:Q", scale=alt.Scale(scheme='blues'), legend=None)
            ).interactive()
            #value.sd
            chart_sd = alt.Chart(scheme).mark_line().encode(
                        x=alt.X(f'{value}'r"\.mean:Q"), 
                        x2=alt.X2(f'{value}'r"\.sd:Q"), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.sd:Q"],
            ).interactive()

        return chart_mean, chart_sd

def plot_creation_single(scheme:object, option:str, value:str, interactive:boolean):
    """
    Parameters
    ----------
    scheme : object
            The dataframe from .csv file
    option : str
            The type of the plot choose by the user
    value : str
            The value to use in the plot
    interactive : boolean
        If the chart will be interactive or not

    Returns
    -------
    chart
        the chart of the value
    """

    title_s = value.capitalize()
    if value == 'encoding_time': title_s = 'Time (seconds)'
    if value == 'encoding_memory': title_s = 'Memory'
    if value == 'feature_vector_size' : title_s = 'Size of feature vector'

    if interactive == True or all(scheme[value]) >= 1.0:
        if option == 'Point':
            #value.point
            chart = alt.Chart(scheme).mark_point(size=100).encode(
                        x=alt.X(value, axis=alt.Axis(title=title_s)), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            ).interactive()

        if option == 'Bar':
            #value.bar
            chart = alt.Chart(scheme).mark_bar().encode(
                        x=alt.X(value, axis=alt.Axis(title=title_s)), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            ).interactive()

    # if the value are between 0 and 1 the plot are not interactive to limit the domain value of the x axes
    if interactive == False or all(scheme[value]) <= 1.0:
        if option == 'Point':
            #value.point
            chart = alt.Chart(scheme).mark_point(size=100).encode(
                        x=alt.X(value, axis=alt.Axis(title=title_s)), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            )

        if option == 'Bar':
            #value.bar
            chart = alt.Chart(scheme).mark_bar().encode(
                        x=alt.X(value, axis=alt.Axis(title=title_s)), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            )

            

    return chart