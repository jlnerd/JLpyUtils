
# coding: utf-8

# In[1]:


#To call in code, add the following command
# import sys
# desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
# sys.path.insert(0, desktop_path+'/JLpy_modules')
# import quick_plots

#normal plot with blue pts
def plot(df,x_label,y_label):
    plt.plot(df[x_label],df[y_label],'ob')
    plt.xlabel(x_label,fontsize = 16)
    plt.ylabel(y_label,fontsize = 16)
    plt.grid(which='major',color = 'dimgray')
    plt.grid(which='minor',color='lightgray')
    plt.tick_params(axis='both',labelsize = 14)

#log log plot
def loglog_plot(df,x_label,y_label):
    plot(df,x_label,y_label)
    plt.xscale("log")
    plt.yscale("log")

#Plot correlation matrix
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns,rotation='vertical');
    plt.yticks(range(len(corr.columns)), corr.columns);
    
#Save this notebook as python version
get_ipython().system('jupyter nbconvert --to script quick_plots.ipynb')

