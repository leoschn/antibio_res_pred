import argparse
import glob
import os
import random
import seaborn as sns
import pandas as pd
import re
import numpy as np
import matplotlib.image as mpimg
import sys

from matplotlib import pyplot as plt

sys.path.append(os.getcwd())



# from build_image import build_image_ms1_wiff_charge_filtered, build_image_ms1_wiff_charge_filtered_apex_only

"""
find . -name '*.mzML' -exec cp -prv '{}' '/home/leo/PycharmProjects/pseudo_image/data/raw_data' ';'
copy des mzml depuis lecteur
"""
antibiotic_tests = ['AMC (disk)','AMK (disk)','AMK (mic)','AMK (vitek)','AMP (vitek)','AMX (disk)',
    'AMX (vitek)','ATM (disk)','ATM (vitek)','CAZ (disk)','CAZ (mic)','CAZ (vitek)','CHL (vitek)','CIP (disk)',
    'CIP (vitek)','COL (mic)','CRO (mic)','CRO (vitek)','CTX (disk)','CTX (mic)','CTX (vitek)',
    'CXM (vitek)','CZA (disk)','CZA (vitek)','CZT (disk)','CZT (vitek)','ETP (disk)','ETP (mic)','ETP (vitek)',
    'FEP (disk)','FEP (mic)','FEP (vitek)','FOS (disk)','FOX (disk)','FOX (vitek)','GEN (disk)','GEN (mic)',
    'GEN (vitek)','IPM (disk)','IPM (mic)','IPM (vitek)','LVX (disk)','LVX (vitek)','MEC (disk)',
    'MEM (disk)','MEM (mic)','MEM (vitek)','NAL (vitek)','NET (disk)','OFX (vitek)','PIP (vitek)','PRL (disk)',
    'SXT (disk)','SXT (vitek)','TCC (disk)','TCC (vitek)','TGC (disk)','TGC (vitek)',
    'TIC (disk)','TIC (vitek)','TOB (disk)','TOB (vitek)','TZP (disk)','TZP (mic)','TZP (vitek)']

antibiotic_enterrobacter_breakpoints = {
    'AMC (disk)': {"S":14, "R":14 },
    'AMK (disk)': {"S":18, "R":18 },
    'AMK (mic)': {"S":8, "R":8 },
    'AMK (vitek)': {"S":8, "R":8 },
    'AMP (vitek)': {"S":8, "R":8 },
    'AMX (disk)': {"S":14, "R":14 },
    'AMX (vitek)': {"S":8, "R":8 },
    'ATM (disk)': {"S":26, "R":21 },
    'ATM (vitek)': {"S":1, "R":4 },
    'CAZ (disk)': {"S":22, "R":22 },
    'CAZ (mic)': {"S":1, "R":4 },
    'CAZ (vitek)': {"S":1, "R":4 },
    'CHL (vitek)': {"S":16, "R":16 },
    'CIP (disk)': {"S":25, "R":22 },
    'CIP (vitek)': {"S":0.25, "R":0.5 },
    'COL (disk)': {"S":None, "R":None }, # : https://academic-oup-com.docelec.univ-lyon1.fr/cid/article/71/9/e523/5735218?login=true&token=eyJhbGciOiJub25lIn0.eyJleHAiOjE3NDU2NjA0NTgsImp0aSI6IjcxYzJmOWI1LTlhMWYtNGRiMy1iYmE0LTA0MGRlMTU3NjdmZSJ9.
    #deleted since method is not accurate (DO NOT USE IT)
    'COL (mic)': {"S":2, "R":2 },
    'CRO (mic)': {"S":1, "R":2 },
    'CRO (vitek)': {"S":1, "R":2 },
    'CTX (disk)': {"S":20, "R":17 },
    'CTX (mic)': {"S":1, "R":2 },
    'CTX (vitek)': {"S":1, "R":2 },
    'CXM (vitek)': {"S":0.001, "R":8 },
    'CZA (disk)': {"S":13, "R":13 },
    'CZA (vitek)': {"S":8, "R":8 },
    'CZT (disk)': {"S":22, "R":22 },
    'CZT (vitek)': {"S":2, "R":2 },
    'ETP (disk)': {"S":23, "R":23 },
    'ETP (mic)': {"S":0.5, "R":0.5 },
    'ETP (vitek)': {"S":0.5, "R":0.5 },
    'FEP (disk)': {"S":27, "R":24 },
    'FEP (mic)': {"S":1, "R":4 },
    'FEP (vitek)': {"S":1, "R":4 },
    'FOS (disk)': {"S":24, "R":24 },#pas clair ?
    'FOX (disk)': {"S":19, "R":19 },#screen only ?
    'FOX (vitek)': {"S":8, "R":8 },#screen only ? high sensitivity but poor specificity for identification of AmpC-producing Enterobacterales
    'GEN (disk)': {"S":17, "R":17 },
    'GEN (mic)': {"S":2, "R":2 }, #entre parenthèse
    'GEN (vitek)': {"S":2, "R":2 }, #entre parenthèse cf https://www.eucast.org/eucastguidancedocuments/ ?
    'IPM (disk)': {"S":22, "R":19 },
    'IPM (mic)': {"S":2, "R":4 },
    'IPM (vitek)': {"S":2, "R":4 },
    'LTM (disk)': {"S":None, "R":None }, # Lactimidomycin ?
    'LVX (disk)': {"S":23, "R":19 },
    'LVX (vitek)': {"S":0.5, "R":1 },
    'MEC (disk)': {"S":15, "R":15 },
    'MEM (disk)': {"S":22, "R":16 },
    'MEM (mic)': {"S":2, "R":8 },
    'MEM (vitek)': {"S":2, "R":8 },
    'NAL (vitek)': {"S":2, "R":8 }, #pas présent dans EUCAST, trouvé dans CLSI M100 (for uninary tract only)
    'NET (disk)': {"S":15, "R":12 }, #insuffisant evidencence for EUCAST, found in CLSI M100
    'OFX (vitek)': {"S":0.25, "R":0.5 },
    'PIP (vitek)': {"S":8, "R":8 },
    'PRL (disk)': {"S":20, "R":20 },
    'SXT (disk)': {"S":14, "R":11 },
    'SXT (vitek)': {"S":2, "R":4 },
    'TCC (disk)': {"S":8, "R":16 },
    'TCC (vitek)': {"S":23, "R":20 },
    'TEM (disk)': {"S":None, "R":None },#Abréviation non standard
    'TEM (vitek)': {"S":None, "R":None },#Abréviation non standard
    'TGC (disk)': {"S":18, "R":18 }, #pour E.coli et C.koseri seulement
    'TGC (vitek)': {"S":0.5, "R":0.5 },
    'TIC (disk)': {"S":13, "R":20 },
    'TIC (vitek)': {"S":8, "R":16 },
    'TOB (disk)': {"S":16, "R":16 }, #entre parenthèse cf https://www.eucast.org/eucastguidancedocuments/ ?
    'TOB (vitek)': {"S":2, "R":2 }, #entre parenthèse cf https://www.eucast.org/eucastguidancedocuments/ ?
    'TZP (disk)': {"S":20, "R":20 },
    'TZP (mic)': {"S":8, "R":8 },
    'TZP (vitek)': {"S":8, "R":8 },
}


def create_antibio_dataset(path='230804_strain_peptides_antibiogram_Enterobacterales.xlsx',base_path=None):
    """
    Extract and build file name corresponding to each sample and transform antioresistance measurements to labels
    :param path: excel path
    :return: dataframe
    """
    base_path = os.getcwd()
    df = pd.read_excel(os.path.join(base_path,path), header=1)
    df = df[['sample_name','species','AMC (disk)','AMK (disk)','AMK (mic)','AMK (vitek)','AMP (vitek)','AMX (disk)',
    'AMX (vitek)','ATM (disk)','ATM (vitek)','CAZ (disk)','CAZ (mic)','CAZ (vitek)','CHL (vitek)','CIP (disk)',
    'CIP (vitek)','COL (mic)','CRO (mic)','CRO (vitek)','CTX (disk)','CTX (mic)','CTX (vitek)',
    'CXM (vitek)','CZA (disk)','CZA (vitek)','CZT (disk)','CZT (vitek)','ETP (disk)','ETP (mic)','ETP (vitek)',
    'FEP (disk)','FEP (mic)','FEP (vitek)','FOS (disk)','FOX (disk)','FOX (vitek)','GEN (disk)','GEN (mic)',
    'GEN (vitek)','IPM (disk)','IPM (mic)','IPM (vitek)','LVX (disk)','LVX (vitek)','MEC (disk)',
    'MEM (disk)','MEM (mic)','MEM (vitek)','NAL (vitek)','NET (disk)','OFX (vitek)','PIP (vitek)','PRL (disk)',
    'SXT (disk)','SXT (vitek)','TCC (disk)','TCC (vitek)','TGC (disk)','TGC (vitek)',
    'TIC (disk)','TIC (vitek)','TOB (disk)','TOB (vitek)','TZP (disk)','TZP (mic)','TZP (vitek)']]

    for test in antibiotic_tests :# S - Susceptible R - Resistant U- Uncertain
        #convert to string and transform (pex >8 to 8)
        df[test] = df[test].map(lambda x :float(str(x).replace('>','').replace('<','')))
        #categorise each antibioresistance according to AST breakpoints table
        df[test+' cat']= 'NA'
        df=df.copy()
        if 'mic' in test or 'vitek' in test :
            try :
                df.loc[df[test] <= antibiotic_enterrobacter_breakpoints[test]['S'], test+ ' cat'] = 'S'
                df.loc[df[test] >= antibiotic_enterrobacter_breakpoints[test]['R'], test + ' cat'] = 'R'
                df.loc[(antibiotic_enterrobacter_breakpoints[test]['S'] < df[test]) & (df[test]  < antibiotic_enterrobacter_breakpoints[test]['R']), test + ' cat'] = 'U'
            except:
                #for empty cells
                pass
        elif 'disk' in test:
            try :
                df.loc[df[test] >= antibiotic_enterrobacter_breakpoints[test]['S'], test + ' cat'] = 'S'
                df.loc[df[test] <= antibiotic_enterrobacter_breakpoints[test]['R'], test + ' cat'] = 'R'
                df.loc[
                    (antibiotic_enterrobacter_breakpoints[test]['S'] > df[test]) & (df[test] > antibiotic_enterrobacter_breakpoints[test][
                        'R']), test + ' cat'] = 'U'
            except:
                pass
    return df


def analyse_antibio(df_path,th):
    df = pd.read_csv(df_path)
    os.makedirs('antibio_stats',exist_ok=True)
    df['reduced_species']=df['species'].apply(lambda x :  x.split(' ')[0]+ ' ' +x.split(' ')[1] if len(x.split(' '))>=2 else x)
    species_grouping = df[['sample_name','reduced_species']].groupby('reduced_species').count()
    main_spe = species_grouping[species_grouping['sample_name']>=th].index.tolist()
    sns.set_theme(rc={'figure.figsize': (32,24)})

    df_main = df[df['reduced_species'].isin(main_spe)]
    palette = {'M':'black','R':'red','U':'orange','S':'green'}
    for antibio_res_type in ['AMC (disk)','AMK (disk)','AMK (mic)','AMK (vitek)','AMP (vitek)','AMX (disk)',
    'AMX (vitek)','ATM (disk)','ATM (vitek)','CAZ (disk)','CAZ (mic)','CAZ (vitek)','CHL (vitek)','CIP (disk)',
    'CIP (vitek)','COL (mic)','CRO (mic)','CRO (vitek)','CTX (disk)','CTX (mic)','CTX (vitek)',
    'CXM (vitek)','CZA (disk)','CZA (vitek)','CZT (disk)','CZT (vitek)','ETP (disk)','ETP (mic)','ETP (vitek)',
    'FEP (disk)','FEP (mic)','FEP (vitek)','FOS (disk)','FOX (disk)','FOX (vitek)','GEN (disk)','GEN (mic)',
    'GEN (vitek)','IPM (disk)','IPM (mic)','IPM (vitek)','LVX (disk)','LVX (vitek)','MEC (disk)',
    'MEM (disk)','MEM (mic)','MEM (vitek)','NAL (vitek)','NET (disk)','OFX (vitek)','PIP (vitek)','PRL (disk)',
    'SXT (disk)','SXT (vitek)','TCC (disk)','TCC (vitek)','TGC (disk)','TGC (vitek)',
    'TIC (disk)','TIC (vitek)','TOB (disk)','TOB (vitek)','TZP (disk)','TZP (mic)','TZP (vitek)']:
        df_antibio = df_main[['sample_name','species',antibio_res_type+' cat']]
        df_antibio = df_antibio.fillna('M')
        sns.countplot(data=df_antibio, x="species", hue=antibio_res_type+' cat',palette=palette)
        plt.title(antibio_res_type)
        plt.xticks(rotation=30)
        plt.savefig('antibio_stats/'+antibio_res_type+'.png')
        plt.clf()

        #GEN(mic) ETP(mic) CRO(mic)



if __name__ =='__main__' :
    df_label = create_antibio_dataset()
    df_label.to_csv('antibiores_labels.csv',index=False)
    df = analyse_antibio('antibiores_labels.csv',20)
