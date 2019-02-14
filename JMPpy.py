
#convert .jmp file to .csv given a file path
def jmp_to_csv(file_path):
    #Load modules
    import pandas as pd
    from win32com.client import Dispatch
    import os
    
    root_path = os.path.split(file_path)[0]
    new_file_name = os.path.splitext(os.path.split(file_path)[1])[0]+'.csv'
    
    new_path = root_path + new_file_name
    
    jmp = Dispatch(r"C:\Program Files\SAS\JMPPRO\14\jmp.exe")    
    doc = jmp.OpenDocument(file_path)    
    
    doc.SaveAs(new_file_name)

    df = pd.read_csv(new_file_name)
    return(df)


# In[3]:


import JMPpy
import os
import codecs
root_path = '//il1-ecsfiler01.apple.com/boober05/Users/John_L/LIV/OM_Fits'
# root_path_folders_files = pd.DataFrame(os.listdir(root_path),columns=['Folders and Files in Root Path'])
# root_path_folders_files
file = '10.18.20118_B_All DAAS Rel Wafer Mapper Data_t0Linked_6,7.95umOnly_V2.jmp'
file_path = root_path + '/'+file
file_path = file_path.replace(" ","\ ")
print(file_path)

df = JMPpy.jmp_to_csv(file_path)


# In[24]:


import subprocess
process = subprocess.Popen([r"C:\Program Files\SAS\JMPPRO\14\jmp.exe",
                            file_path])


# In[10]:


file_path

