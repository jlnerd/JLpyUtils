
# coding: utf-8

# In[5]:


#To call in code, add the following command
# import sys, os
# desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
# sys.path.insert(0, desktop_path+'/JLpy_Utilities')
# import quick_plots as qp

#Save the jupyter notebook file as .py to load as module in other notebooks
try:
    get_ipython().system('jupyter nbconvert --to script quick_plots.ipynb')
except:
    print('')

#Load necessary modules
import matplotlib.pyplot as plt

#normal plot with blue pts
def plot(df,x_label,y_label,color='b'):
    """
    simple x-y plot
    """
    
    plt.plot(df[x_label],df[y_label],'o',color=color)
    plt.xlabel(x_label,fontsize = 16)
    plt.ylabel(y_label,fontsize = 16)
    plt.grid(which='major',color = 'dimgray')
    plt.grid(which='minor',color='lightgray')
    plt.tick_params(axis='both',labelsize = 14)
    
def plot_w_legend(df,x_label,y_label,legend_label='None'):
    """
    Description:
        simple x y plot with legend. The legend and groups of data are produced by finding non-duplicate labels from the legend label column, then plotting for each unique legend label group.
    """
  
    #Build Legend list
    list_legend_values = df[legend_label].drop_duplicates().reset_index(drop=True)
    
    #Build Color list
    import matplotlib.cm as cm
    import numpy as np
    colors = cm.rainbow(np.linspace(0, 1, len(list_legend_values)))
    
    for i in range(len(list_legend_values)):
        legend_value = list_legend_values[i]
        color = colors[i]
        df_subset = df[df[legend_label]==legend_value]
        
        plot(df_subset,x_label,y_label,color)
                
    plt.legend(list_legend_values)

#log log plot
def loglog_plot(df,x_label,y_label):
    """
    simple log-log plot
    """
    plot(df,x_label,y_label)
    plt.xscale("log")
    plt.yscale("log")

#Plot correlation matrix
def plot_corr(df,size=10):
    '''
    Desription:
        Function plots a graphical correlation matrix for each pair of columns in the dataframe.
    Inputs:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    Returns:
        df_correlations    
    '''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns,rotation='vertical');
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    #Build Correlation df
    df_correlations = df.corr()
    
    return df_correlations
    
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

