import matplotlib as mpl

#https://matplotlib.org/users/customizing.html
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['grid.color'] =  'lightgray'
mpl.rcParams['grid.linestyle'] = '-'
mpl.rcParams['grid.linewidth'] = 1
mpl.rcParams['axes.grid.which'] = 'both'
mpl.rcParams['axes.grid']=True 
mpl.rcParams['xtick.minor.visible']=True
mpl.rcParams['ytick.minor.visible']=True
mpl.rcParams['xtick.top']=True
mpl.rcParams['ytick.right']=True
mpl.rcParams['xtick.direction']='inout'
mpl.rcParams['ytick.direction']='inout'
mpl.rcParams['figure.dpi']= 150

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick

import numpy as np
import pandas as pd

def apply_common_plot_format(x_label,y_label,title_string,
                            legend_lines,legend_labels,legend_title, legend_anchor_pt = 'best'
                           ):
    """
    Apply common formating options to plots
    
    Arguments: 
        x_label, y_label,
        title_string: title of plot
        legend lines: [Line2D([0], [0], color=c, lw=1)]
        legend_labels: [name]
        legend_title: title of legend
        legend_anchor_pt: 'best', (1,1), or some other (x,y) coordinate
    Examples:
        df_group = df.groupby(group_label)
        legend_labels = df[group_label].drop_duplicates()
        legend_labels = [ '%.2f' % label for label in legend_labels ] # round to 2 digits
        legend_lines = []
        colors = cm.rainbow(np.linspace(0, 1, len(legend_labels)))
        i=0
        for name, group in df_group:
            c = colors[i]
            plt.plot(group[x_label], group[y_label],color = c, linestyle='-')
            legend_lines.append(Line2D([0], [0], color=c, lw=1)) 
            i = i+1
        title_string = 'title'
        legend_title = 'legend title'
        apply_common_plot_format(x_label,y_label,title_string, legend_lines, legend_labels, legend_title,legend_anchor_pt=(1,1))

    """
    
    plt.xlabel(x_label,fontsize=16)
    plt.ylabel(y_label,fontsize=16)
    plt.title(title_string,fontsize=16)
    plt.grid(which='major',color = 'darkgray')
    plt.grid(which='minor',color='lightgray')
    plt.tick_params(axis='both',labelsize = 14)
    if legend_anchor_pt == 'best':
        legend = plt.legend(legend_lines, legend_labels,fontsize=14,title = legend_title)
    else:
        legend = plt.legend(legend_lines, legend_labels,fontsize=14,title = legend_title, bbox_to_anchor=legend_anchor_pt)
    plt.setp(legend.get_title(),fontsize=12)


def plot_corr_and_pareto(df,label,size=10):
    '''
    Description: 
        Plt correlation matrix for entire data frame, then plot pareto bar-chart for 1 label of interest
    Inputs:
        df: pandas DataFrame
        label: column/header for which you want to plot the bar-chart pareto for
        size: vertical and horizontal size correlation chart
    Returns:
        df_correlations, df_label_pareto, df_label_pareto_sorted
        '''
    
    #plot correlation chart
    df_correlations = plot_corr(df,size)
    plt.show()
    
    #Fetch pareto for selected label
    df_label_pareto = df_correlations[label]
    df_label_pareto_sorted = df_correlations[label].sort_values(ascending=False)

    plt.bar(df_label_pareto_sorted.index,df_label_pareto_sorted)
    plt.xticks(rotation = 'vertical')
    plt.ylabel(label+" Correlation Factor",fontsize = 14)
    plt.title(label+" Correlation Factor Pareto", fontsize = 14)
    plt.tick_params(axis='both',labelsize = 14)
    plt.show()
    
    return df_correlations, df_label_pareto, df_label_pareto_sorted

def plot_by_group_subgroup(df,
                           x_label,y_label,
                           group_label,sub_group_label,
                           title_string,
                           xscale = 'linear', yscale = 'linear', **kwargs):
                           
    df_group = df.groupby(by=group_label)
    
    colors = iter(cm.rainbow(np.linspace(0,1,len(df_group))))
    
    legend_labels = []
    legend_lines = []
    
    for group_ID, group_subset in df_group:
        c = next(colors)
        legend_lines.append(Line2D([0], [0], color=c, lw=1)) 
        legend_labels.append(group_ID)
        df_sub_group = group_subset.groupby(sub_group_label)

        for sub_group_ID, sub_group_subset in df_sub_group:
            plt.plot(sub_group_subset[x_label],sub_group_subset[y_label],color = c, **kwargs)
            plt.xscale(xscale)
            plt.yscale(yscale)

    legend_title = group_label
    apply_common_plot_format(x_label,y_label,title_string, legend_lines, legend_labels, legend_title,legend_anchor_pt=(1,1))
