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
#import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--deepmd","-d",help="deepmd path file")
parser.add_argument("--inputfile","-if",help="input files for vasp cal, default is cwd+inputs, if input, please input the absolute path")
parser.add_argument("--temperature","-t",default=4000,type=int,help="simulation temperature")
parser.add_argument("--step","-s",default=1,type=int,help="step")
parser.add_argument("--range","-r",type=str,help="0-2, means from 0 to 2, default is for all folders")
parser.add_argument("--recal_dir_name","-rd",default='recal',help="recal directory name")
# parser.add_argument("--run_vasp","-rv",help="run vasp?, default without input is Yes")
# parser.add_argument("--sub_command","-sc", default='/u/systems/UGE8.6.4/bin/lx-amd64/qsub',help="job submission command: default is /u/systems/UGE8.6.4/bin/lx-amd64/qsub")

args   = parser.parse_args()
temp = args.temperature

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
                content = f.read().replace('temperature', str(temp))
                sigma_value = 8.6173303e-5 * temp
                content = content.replace('sigma_value', f'{sigma_value:.4f}')
            tmp_incar_path = os.path.join(inputfile,'INCAR_tmp')
            with open(tmp_incar_path, 'w') as f:
                f.write(content)
            copy(tmp_incar_path,os.path.join(target_path,'INCAR'))
            copy(os.path.join(inputfile, 'KPOINTS'), target_path)
            os.symlink(os.path.join(inputfile, 'POTCAR'), os.path.join(target_path, 'POTCAR'))
            # if run_vasp:
            #     print("run vasp",target_path)
            #     run(cwd,target_path)
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
recal_path = os.path.join(os.getcwd(), args.recal_dir_name)
try:
    os.mkdir(recal_path)     
except:
    print('***recal exists in',recal_path)

lmp2pos(ls,sel_nsw,copybs = True)
    

