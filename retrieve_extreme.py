import pandas as pd, glob,os, shutil

ids=pd.read_json('ext_max.json')


path = '/dhc/groups/mpws2022cl1/images/heart/png/50000_RGB_0-16-39/'+ '**/*.png'      
r2=r'PCA_extreme_values'
# string to search for

# empty list to hold files that contain matching string
files_to_check = []

# looping through all the filenames returned
# set recursive = True to look in sub-directories too
#for i in glob.iglob(path, recursive=True):
    # adding error handling just in case!
#    try:
#        with open(i) as f:
#            # read the file as a string
#            contents = f.read()
            # if the search term is found append to the list of files
#            if(ids in contents):
#                files_to_check.append(filename)
#    except:
#        pass


# ''' 
#for i in ids:
#  esc_set = str(ims) + glob.escape(str(root))  + "*.png"
#  
#  #for py in (glob.glob(esc_set)):
#  print(esc_set)

#'''
#for i in ids:
    #l = glob.glob(path+ str(i)+'*/.png')
    #csv_max_files = glob.glob(path+'*max*.{}'.format('csv'))
#csv_max_files
    #ll = list(set(sorted(glob.glob(os.path.join(root, str(i)+'_*.png')))))
   # print(l)



files = []
for a in ids:
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and i.startswith(ids[a]):
            files.append(i)
print(files_to_check)