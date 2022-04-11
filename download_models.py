import gdown
import os


# link = "https://drive.google.com/file/d/1-wLhzIjEKqXtNZaIkz84syF_ufjC2yhV/"
ffhq_model = "https://drive.google.com/u/0/uc?id=1-wLhzIjEKqXtNZaIkz84syF_ufjC2yhV&export=download"
restyle_e4e_ffhq_encode = "https://drive.google.com/u/0/uc?id=1-0TEycjG6t7ZYjRGf203McnLhBtu9KB6&export=download"
restyle_psp_ffhq_encode = "https://drive.google.com/u/0/uc?id=1-2sU-fdcIjDbrxfaZRWcoWRFx8in5WOG&export=download"

all_links = [ffhq_model, restyle_e4e_ffhq_encode, restyle_psp_ffhq_encode]

# folder_name = 'models1'

# if not os.path.exists(folder_name):
#     os.makedirs(folder_name)

for x in all_links:   
    gdown.download(x)