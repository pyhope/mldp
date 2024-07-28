#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 08:37:37 2020

@author: jiedeng
"""
from dpdata import LabeledSystem
import numpy as np
import os

def extract_lammps_log(logfile,skip_header,skip_footer):
    """
    extract log file, detailed col info info may vary
    """
    out = np.genfromtxt(logfile,comments='WARNING',
                          skip_header=skip_header,skip_footer=skip_footer)
    return out

def extract_outcar(outcar):
    """
    extract e, f, v
    """
    ### get confgis that were recalculated
    ls     = LabeledSystem(outcar,fmt='outcar')
    fp     = open(outcar)
    fp.readline()
    nsw_sel = fp.readline()
    if 'nsw_sel' in nsw_sel:
        print('file generated by merge_out.py')
        tmp     = nsw_sel.split('=')[1].strip().split(' ')
        nsw_sel = [int(tmp_idx) for tmp_idx in tmp]
    ### get confgis that were recalculated    
    etot   = ls['energies']
    nsw    = np.array(nsw_sel).astype(int) -1 # relative nsw, starting from 0
    stress = ls['virials']
    forces = ls['forces']
    return etot, stress, forces, nsw

def extract_nn_pred(test_folder,prefix):
    """
    extract e, f, v
    """
    es_dirs = []; fs_dirs = []; vs_dirs = [];  
    if type(prefix) is str:
        prefixs = [prefix]
    else:
        prefixs = prefix

    for pref in prefixs:
       e_dir, f_dir, v_dir = _make_dir(test_folder, pref)
       es_dirs.append(e_dir)
       fs_dirs.append(f_dir)
       vs_dirs.append(v_dir)
    
    es = []
    fs = []
    vs = []    

    for dire in es_dirs:
        tmp = np.loadtxt(dire)
        es.append(tmp[:,1])
    es = np.array(es)
    nmodels, nframes = es.shape

    for dire in fs_dirs:
        tmp = np.loadtxt(dire)
        tmp = tmp[:,3:6]
        natoms = tmp.shape[0]//nframes
        yy =tmp.reshape((nframes,natoms,3))
        zz = yy.reshape((nframes,natoms*3))
        fs.append(zz)    
    fs = np.array(fs)
    for dire in vs_dirs:
        tmp = np.loadtxt(dire)
        tmp = tmp[:,9:]
        vs.append(tmp)   
    vs = np.array(vs)
    return es,fs,vs, nframes, natoms

def extract_org_nn_pred(test_folder,prefix):
    """
    extract e, f, v
    TODO: natoms should be self contained
    """
    es_dirs = []; fs_dirs = []; vs_dirs = [];  
    if type(prefix) is str:
        prefixs = [prefix]
    else:
        prefixs = prefix

    for pref in prefixs:
       e_dir, f_dir, v_dir = _make_dir(test_folder, pref)
       es_dirs.append(e_dir)
       fs_dirs.append(f_dir)
       vs_dirs.append(v_dir)
    
    es = [];es_org = []
    fs = [];fs_org = []
    vs = [];vs_org = []   

    for dire in es_dirs:
        tmp = np.loadtxt(dire)
        es.append(tmp[:,1])
        es_org.append(tmp[:,0])
    es = np.array(es)
    es_org = np.array(es_org)
    nmodels, nframes = es.shape

    for dire in fs_dirs:
        tmp = np.loadtxt(dire)
        tmp2, tmp1  = tmp[:,:3], tmp[:,3:6]
#        tmp2 = tmp[:,0:3]        
        natoms = tmp.shape[0]//nframes
        
        yy  = tmp1.reshape((nframes,natoms,3))
        zz  = yy.reshape((nframes,natoms*3))
        yy2 = tmp2.reshape((nframes,natoms,3))
        zz2 = yy2.reshape((nframes,natoms*3))
        fs.append(zz)   
        fs_org.append(zz2)
    fs = np.array(fs)
    fs_org = np.array(fs_org)
    for dire in vs_dirs:
        tmp = np.loadtxt(dire)
        tmp2, tmp1 = tmp[:,:9], tmp[:,9:]
#        tmp2 = tmp[:,:9]
        vs.append(tmp1)   
        vs_org.append(tmp2) 
    vs = np.array(vs)
    vs_org = np.array(vs_org)
    return es_org,fs_org,vs_org,es,fs,vs, nframes, natoms


def _make_dir(test_folder,prefix):
    e_dir = os.path.join(test_folder,prefix+'.e.out')
    f_dir = os.path.join(test_folder,prefix+'.f.out')
    v_dir = os.path.join(test_folder,prefix+'.v.out')
    return e_dir, f_dir, v_dir
    
def dev_nn(es,fs,vs,natom=160):
    """
    model deviation for nn results
    """
    var_e = np.var(es,axis=0);std_e = np.std(es,axis=0)
    var_f = np.var(fs,axis=0);std_f = np.std(fs,axis=0) 
    max_var_f = np.max(var_f,axis=1);max_std_f = np.max(std_f,axis=1)
    var_v = np.var(vs,axis=0);std_v = np.std(vs,axis=0);
    max_var_v = np.max(var_v,axis=1);max_std_v = np.max(std_v,axis=1)
    
    
    dfs = fs - np.mean(fs,axis=0)
    
    dpgen_fs                = np.sqrt(np.sum(dfs**2,axis=0))
    dpgen_fs_per_atom       = dpgen_fs.reshape((len(dpgen_fs),natom,3))
    mean_dpgen_fs_per_atom  = np.sqrt(np.sum(dpgen_fs_per_atom,axis=2))
    max_dpgen_fs            = np.max(mean_dpgen_fs_per_atom,axis=1)
    mean_std_f              = np.mean(std_f,axis=1)

    return max_dpgen_fs, mean_std_f

def dev_vasp_nn(vasp,nn,natoms=160):
    """
    compare vasp and nn results
    vasp results format:
    e : step
    f : step*natom*3
    v : step*3*3
    nn results fomat:
    es : step
    fs : step*(natoms*3)
    vs : step*9
    """
    
    #root-mean-square deviation
#    assert not(type(nn['e'][0]) is float)
    
    rmsd_e= abs(vasp[0] - nn[0]) # each step, there is 1 energy for the whole system
    # each step, there are natoms*3 forces
    vasp_force = vasp[1].reshape((len(vasp[1]),natoms*3))
    rmsd_f=_rmse(vasp_force,nn[1])
    vasp_virial= vasp[2].reshape((len(vasp[2]),3*3))
    rmsd_v=_rmse(vasp_virial,nn[2])
    return rmsd_e,rmsd_f,rmsd_v

def _rmse(x1,x2):
    """
    calculate the root mean squre error
    """
    dx = x1-x2
    rmse = np.sqrt(np.sum(dx**2,axis=1)/dx.shape[1])
    return rmse

#_rmse(vasp[0],nn[0])
#
#
#ff = fs[0]
#ff2 = ff+1
#
#_rmse(ff,ff2)

