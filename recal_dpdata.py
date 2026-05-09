#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:35:11 2020

generate poscar from deepmd files

if recal exist, check if the NSW is selected?
@author: jiedeng
"""
import os
from shutil import copy
#from shared_functions import load_paths
import dpdata
import numpy as np
#import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--deepmd","-d",help="deepmd path file")
parser.add_argument("--inputfile","-if",help="input files for vasp cal, default is cwd+inputs, if input, please input the absolute path")
parser.add_argument("--temperature","-t",default=None,type=int,help="simulation temperature")
parser.add_argument("--step","-s",default=1,type=int,help="step")
parser.add_argument("--range","-r",type=str,help="0-2, means from 0 to 2, default is for all folders")
parser.add_argument("--recal_dir_name","-rd",default='recal',help="Path to the recal directory")
parser.add_argument("--spin","-sp",action='store_true',help="add magmom and nupdown in INCAR, default is False")
parser.add_argument("--Fe_mag_moment","-fmm",default=2,type=int,help="Fe magnetic moment, default is 2")
parser.add_argument("--potcar_by_elements","-pe",action='store_true',help="generate POTCAR by elements, default is False")
parser.add_argument("--calc_nband_from_nelec","-cn",action='store_true',help="calculate Nband from Nelec in POSCAR, default is False")

# parser.add_argument("--run_vasp","-rv",help="run vasp?, default without input is Yes")
# parser.add_argument("--sub_command","-sc", default='/u/systems/UGE8.6.4/bin/lx-amd64/qsub',help="job submission command: default is /u/systems/UGE8.6.4/bin/lx-amd64/qsub")

args   = parser.parse_args()
temp = args.temperature
elem_dict = {'Fe':14,'Mg':8,'O':6,'W':12, 'Pt':10}

if temp is None:
    print('***temperature is not provided, read from set.000/fparam.npy')
    fparam = np.load(os.path.join(args.deepmd,'set.000','fparam.npy'))
    temp = float(fparam.flatten()[0]) / 8.6173303e-5
    print('***temperature read from fparam.npy is',temp)

def lmp2pos(ls,sel_nsw,copybs=False):
    """
    build POSCAR based on deepmd given in the path
    copy INCAR, POTCAR, KPOINTS into the same folder
    
    """
#    lmps1=glob.glob(os.path.join(path,'*lammps*'))
#    lmps2=glob.glob(os.path.join(path,'*dump*'))
#    if lmps2 is []:
#        print("No dump file found")
#    elif not(lmps2 is []):
#        lmp = lmps2[0]    
#    elif not(lmps1 is []):
#        lmp = lmps1[0]
#    else:
#        print("No dump file and lammps file found")
        
#    path = os.path.join(path,'recal')

#    ls=dpdata.System(lmp,fmt='lammps/dump')
#    recal_path = os
    if  sel_nsw is None:
        sel_nsw = range(0,len(ls),args.step)       
    else:
        sel_nsw = sel_nsw
    for i in sel_nsw:
        print(i)
        if os.path.exists(os.path.join(recal_path,str(i+1))):
            print("Folder {0} already exists,skip making".format(i))
        else:
            os.mkdir(os.path.join(recal_path,str(i+1))) # return None
            target_path    = os.path.join(recal_path,str(i+1))                   
            ls.to_vasp_poscar(os.path.join(target_path,'POSCAR'),frame_idx=i)
            with open(os.path.join(inputfile,'INCAR')) as f:
                content = f.read().replace('temp_value', f'{temp:.1f}')
                sigma_value = 8.6173303e-5 * temp
                content = content.replace('sigma_value', f'{sigma_value:.4f}')
            tmp_incar_path = os.path.join(inputfile,'INCAR_tmp')
            with open(tmp_incar_path, 'w') as f:
                f.write(content)
            copy(tmp_incar_path,os.path.join(target_path,'INCAR'))
            copy(os.path.join(inputfile, 'KPOINTS'), target_path)
            
            incar_path = os.path.join(target_path, "INCAR")
            poscar_path = os.path.join(target_path, "POSCAR")
            with open(poscar_path, "r") as f:
                lines = f.readlines()
            
            elements = lines[5].split()
            atoms_per_species = lines[6].split()
            potcar_path = os.path.join(target_path, "POTCAR")
            if args.potcar_by_elements:
                with open(potcar_path, "w") as potcar_file:
                    for element in elements:
                        element_potcar_path = os.path.join(inputfile, "POTCAR_" + element)
                        with open(element_potcar_path, "r") as element_potcar_file:
                            potcar_file.write(element_potcar_file.read())
            else:
                os.symlink(os.path.join(inputfile, 'POTCAR'), potcar_path)

            if args.calc_nband_from_nelec:
                nelec = 0
                for element, count in zip(elements, atoms_per_species):
                    nelec += elem_dict[element] * int(count)
                nband = int(nelec * 3 / 4)
                
                with open(incar_path, "r") as f:
                    incar_content = f.read()
                incar_content = incar_content.replace("nband_value", str(nband))
                with open(incar_path, "w") as f:
                    f.write(incar_content)

            if args.spin:
                Fe, Mg, Si, O = map(int, atoms_per_species)

                magmom = f"{Fe}*{args.Fe_mag_moment} {Mg}*0 {Si}*0 {O}*0"
                nupdw = Fe * args.Fe_mag_moment + Mg * 0 + Si * 0 + O * 0

                with open(incar_path, "r") as f:
                    incar_content = f.read()

                incar_content = incar_content.replace("magmom_value", magmom)
                incar_content = incar_content.replace("nupdown_value", str(nupdw))

                with open(incar_path, "w") as f:
                    f.write(incar_content)

    os.remove(tmp_incar_path)

# from subprocess import call
# def run(cwd,target_path):
#     os.chdir(target_path)
#     sub_file = os.path.join(inputfile,'sub_vasp.sh')
#     call("{1} {0}".format(sub_file, args.sub_command), shell=True)
# #    call("bash {0}".format(sub_file), shell=True)
#     os.chdir(cwd)
    
# print(args.run_vasp)
cwd    = os.getcwd()
if args.deepmd:
    print("Check files in {0}  ".format(args.deepmd))
#    inputpath = args.inputpath
#    paths = load_paths(inputpath)
    ls = dpdata.System(args.deepmd,fmt='deepmd/npy')
else:
    print("deepmd path is not provided .")
    ls = dpdata.System('.',fmt='deepmd/npy')

if args.inputfile:
    print("Check files in {0}  ".format(args.inputfile))
    inputfile = args.inputfile
else:
    print("No folders point are provided. Use default value folders")
    inputfile = os.path.join(cwd,'inputs')

# change inputfile to absolute path, chdir occurs when submitting job

inputfile = os.path.abspath(inputfile)
sel_nsw = None
if args.range:
    tmp     = [int(i) for i in args.range.split('-')]
    sel_nsw = range(tmp[0],tmp[1],args.step)
# run_vasp = True
# if args.run_vasp:
#     run_vasp = False
   
#for path in paths:
#    print('###',path)
recal_path = args.recal_dir_name
try:
    os.makedirs(recal_path)     
except:
    print('***recal exists in',recal_path)

lmp2pos(ls,sel_nsw,copybs = True)
    

