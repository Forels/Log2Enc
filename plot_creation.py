import altair as alt

def plot_creation_single(scheme:object, option:str, value:str, title:str, interactive:bool):
    """
    Parameters
    ----------
    scheme : object
            The dataframe from .csv file
    option : str
            The type of the plot choose by the user
    value : str
            The value to use in the plot
    title : str
            The label of x axis 
    interactive : boolean
        If the chart will be interactive or not

    Returns
    -------
    chart
        the chart of the value
    """

    if interactive == True:
        if option == 'Point':
            #value.point
            chart = alt.Chart(scheme).mark_point(size=100).encode(
                        x=alt.X(value, axis=alt.Axis(title=title)), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            ).interactive()

        if option == 'Bar':
            #value.bar
            chart = alt.Chart(scheme).mark_bar().encode(
                        x=alt.X(value, axis=alt.Axis(title=title)), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            ).interactive()

    # if the value are between 0 and 1 the plot are not interactive to limit the domain value of the x axes
    if all(scheme[f'{value}']) <= 1.0:
        if option == 'Point':
            #value.point
            chart = alt.Chart(scheme).mark_point(size=100).encode(
                        x=alt.X(value, axis=alt.Axis(title=title), scale=alt.Scale(domain=(0, 1.0))), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            )

        if option == 'Bar':
            #value.bar
            chart = alt.Chart(scheme).mark_bar().encode(
                        x=alt.X(value, axis=alt.Axis(title=title), scale=alt.Scale(domain=(0, 1.0))), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            )

    if interactive is False:
        if option == 'Point':
            #value.point
            chart = alt.Chart(scheme).mark_point(size=100).encode(
                        x=alt.X(value, axis=alt.Axis(title=title)), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            )

        if option == 'Bar':
            #value.bar
            chart = alt.Chart(scheme).mark_bar().encode(
                        x=alt.X(value, axis=alt.Axis(title=title)), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', value],
                        color=alt.Color(value, scale=alt.Scale(scheme='blues'), legend=None)
            )

    return chart

def plot_creation(scheme:object, option:str, value:str, xlabel:str):
    """
    Parameters
    ----------
    scheme : object
            The dataframe from .csv file
    option : str
            The type of the plot choose by the user
    value : str
            The value to use in the plot
    xlabel : str
            The label of x axis
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
                        x=alt.X(f'{value}'r"\.mean:Q", axis=alt.Axis(title=xlabel), scale=alt.Scale(domain=(0, 1.0))), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.mean:Q"],
                        color=alt.Color(f'{value}'r"\.mean:Q", scale=alt.Scale(scheme='blues'), legend=None)
            )
            #value.sd
            chart_sd = alt.Chart(scheme).mark_line().encode(
                        x=alt.X(f'{value}'r"\.mean:Q", scale=alt.Scale(domain=(0, 1.0))), 
                        x2=alt.X2(f'{value}'r"\.sd:Q"), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.sd:Q"],
            )

        if option == 'Bar':
            #value.mean
            chart_mean = alt.Chart(scheme).mark_bar().encode(
                        x=alt.X(f'{value}'r"\.mean:Q", axis=alt.Axis(title=xlabel), scale=alt.Scale(domain=(0, 1.0))), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.mean:Q"],
                        color=alt.Color(f'{value}'r"\.mean:Q", scale=alt.Scale(scheme='blues'), legend=None)
            )
            #value.sd
            chart_sd = alt.Chart(scheme).mark_rule().encode(
                        x=alt.X(f'{value}'r"\.mean:Q", scale=alt.Scale(domain=(0, 1.0))), 
                        x2=alt.X2(f'{value}'r"\.sd:Q"), 
                        y=alt.Y("encoding", axis=alt.Axis(title="Encoding methods")),
                        tooltip=['encoding', f'{value}'r"\.sd:Q"]
            )

        return chart_mean, chart_sd

    if all(scheme[f'{value}.mean']) >= 1.0:
    
        if option == 'Point':
            #value.mean
            chart_mean = alt.Chart(scheme).mark_point(size=100).encode(
                        x=alt.X(f'{value}'r"\.mean:Q", axis=alt.Axis(title=xlabel)), 
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
                        x=alt.X(f'{value}'r"\.mean:Q", axis=alt.Axis(title=xlabel)), 
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