import matplotlib as mpl
import matplotlib.pyplot as plt

#https://matplotlib.org/users/customizing.html

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
#mpl.rcParams['figure.dpi']= 100
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.facecolor'] = 'w'

import numpy as np
import pandas as pd

def make_independant_legend(legend_lines,legened_labels,legend_title):
    plt.legend(legend_lines,legened_labels,title=legend_title)
    plt.grid(which='both')
    plt.axis('off')
    plt.tight_layout(rect=(0,0,.3,.3))
    plt.show()

def fetch_color_map_for_primary_color(primary_color, n_colors):
    if primary_color == 'R':
        color_map = plt.cm.hot(np.linspace(0.1,0.7,n_colors))
    elif primary_color == 'G':
        color_map = plt.cm.nipy_spectral(np.linspace(0.4,0.6,n_colors))
    elif primary_color == 'B':
        color_map = plt.cm.jet(np.linspace(0,0.3,n_colors))
    return color_map

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
        df_grp = df.groupby(legend_axis_label)
        legend_labels = list(df[legend_axis_label].unique())
        legend_lines = []
        colors = cm.rainbow(np.linspace(0, 1, len(legend_labels)))
        i=0
        for grp_ID, grp_subset in df_group:
            c = colors[i]
            legend_lines.append(Line2D([0], [0], color=c, lw=1)) 
            plt.plot(grp_subset[x_label], grp_subset[y_label],color = c, linestyle='-')
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
    
    colors = iter(plt.cm.rainbow(np.linspace(0,1,len(df_group))))
    
    legend_labels = []
    legend_lines = []
    
    for group_ID, group_subset in df_group:
        c = next(colors)
        legend_lines.append(mpl.lines.Line2D([0], [0], color=c, lw=1)) 
        legend_labels.append(group_ID)
        df_sub_group = group_subset.groupby(sub_group_label)

        for sub_group_ID, sub_group_subset in df_sub_group:
            sub_group_subset = sub_group_subset.sort_values(x_label)
            plt.plot(sub_group_subset[x_label],sub_group_subset[y_label],color = c, **kwargs)
            plt.xscale(xscale)
            plt.yscale(yscale)

    legend_title = group_label
    apply_common_plot_format(x_label,y_label,title_string, legend_lines, legend_labels, legend_title,legend_anchor_pt=(1,1))
