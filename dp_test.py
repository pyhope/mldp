#!/usr/bin/env python3

#import re
#import os
#import sys
#import argparse
import numpy as np
try:
    from deepmd.Data import DataSets
except:
    from deepmd.utils.data import DeepmdData
#from deepmd.Data import DeepmdData
#from deepmd import DeepEval
try:
    from deepmd import DeepPot
except:
    from deepmd.infer import DeepPot
#from deepmd import DeepDipole
#from deepmd import DeepPolar
#from deepmd import DeepWFC
#from tensorflow.python.framework import ops
import os

try:
    DataSets
except NameError:
    DataSets = None


def _set_name(set_dir):
    return os.path.basename(str(set_dir).rstrip(os.sep))


def _maybe_call(obj, name, default=None):
    attr = getattr(obj, name, default)
    return attr() if callable(attr) else attr


def _keep_sets(data, set_names):
    """Restrict a DeepmdData object to selected set.* directories."""
    if DataSets is not None or set_names is None:
        return data
    keep = [ii for ii in data.dirs if _set_name(ii) in set_names]
    if keep:
        data.dirs = keep
        frames_list = [data._get_nframes(set_name) for set_name in data.dirs]
        data.nframes = np.sum(frames_list)
        data.prefix_sum = np.cumsum(frames_list).tolist()
        data.set_count = 0
        data.iterator = 0
        for attr in ("test_set", "batch_set"):
            if hasattr(data, attr):
                delattr(data, attr)
    return data


def _new_data(system, set_prefix, shuffle_test, dp=None, set_names=None):
    if DataSets is not None:
        return DataSets(system, set_prefix, shuffle_test=shuffle_test)
    type_map = _maybe_call(dp, "get_type_map") if dp is not None else None
    data = DeepmdData(
        system,
        set_prefix=set_prefix,
        shuffle_test=shuffle_test,
        type_map=type_map,
        sort_atoms=False,
    )
    return _keep_sets(data, set_names)


def _add_model_data(data, dp):
    """Register labels and model inputs required by the updated DeepmdData API."""
    if DataSets is not None:
        return data
    has_default_fparam = _maybe_call(dp, "has_default_fparam", False)
    data.add("energy", 1, atomic=False, must=False, high_prec=True)
    data.add("force", 3, atomic=True, must=False, high_prec=False)
    data.add("virial", 9, atomic=False, must=False, high_prec=False)
    if _maybe_call(dp, "has_efield", False):
        data.add("efield", 3, atomic=True, must=True, high_prec=False)
    if dp.get_dim_fparam() > 0:
        data.add(
            "fparam",
            dp.get_dim_fparam(),
            atomic=False,
            must=not has_default_fparam,
            high_prec=False,
        )
    if dp.get_dim_aparam() > 0:
        data.add("aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False)
    if _maybe_call(dp, "has_spin", False):
        data.add("spin", 3, atomic=True, must=True, high_prec=False)
    return data


def _load_eval_input(data, dp, source, numb_frames):
    coord = source["coord"][:numb_frames].reshape([numb_frames, -1])
    box = None if getattr(data, "pbc", True) is False else source["box"][:numb_frames]
    mixed_type = bool(getattr(data, "mixed_type", False))
    if mixed_type:
        atype = source["type"][:numb_frames].reshape([numb_frames, -1])
    else:
        atype = source["type"][0]

    if dp.get_dim_fparam() > 0 and source.get("find_fparam", 0.0) != 0.0:
        fparam = source["fparam"][:numb_frames]
    else:
        fparam = None
    if dp.get_dim_aparam() > 0:
        aparam = source["aparam"][:numb_frames]
    else:
        aparam = None

    extra = {} if DataSets is not None else {"mixed_type": mixed_type}
    if _maybe_call(dp, "has_efield", False):
        extra["efield"] = source["efield"][:numb_frames].reshape([numb_frames, -1])
    if _maybe_call(dp, "has_spin", False):
        extra["spin"] = source["spin"][:numb_frames].reshape([numb_frames, -1])
    return coord, box, atype, fparam, aparam, extra


def _first_fparam(fparam):
    if fparam is None:
        return None
    return fparam.reshape([fparam.shape[0], -1])[0][0]

def l2err (diff) :    
    return np.sqrt(np.average (diff*diff))

def test_ener (args) :
    """
    modify based on args file
    """
    if args['rand_seed'] is not None :
        np.random.seed(args['rand_seed'] % (2**32))

    dp        = DeepPot(args['model'])
    data = _new_data (args['system'], args['set_prefix'], args['shuffle_test'], dp=dp, set_names={"set.001"})
    _add_model_data(data, dp)
    test_data = data.get_test ()
    numb_test = args['numb_test']
    natoms    = len(test_data["type"][0])
    nframes   = test_data["coord"].shape[0]
    numb_test = min(nframes, numb_test)
    coord, box, atype, fparam, aparam, extra = _load_eval_input(data, dp, test_data, numb_test)
    detail_file = args['detail_file']
    if detail_file is not None:
        atomic = True
    else:
        atomic = False

    ret = dp.eval(coord, box, atype, fparam = fparam, aparam = aparam, atomic = atomic, **extra)
    energy = ret[0]
    force  = ret[1]
    virial = ret[2]
    energy = energy.reshape([numb_test,1])
    force  = force.reshape([numb_test,-1])
    virial = virial.reshape([numb_test,9])
    if atomic:
        ae = ret[3]
        av = ret[4]
        ae = ae.reshape([numb_test,-1])
        av = av.reshape([numb_test,-1])

    l2e = (l2err (energy - test_data["energy"][:numb_test].reshape([-1,1])))
    l2f = (l2err (force  - test_data["force"] [:numb_test]))
    l2v = (l2err (virial - test_data["virial"][:numb_test]))
    l2ea = l2e/natoms
    l2va = l2v/natoms

    # print ("# energies: %s" % energy)
    print ("# number of test data : %d " % numb_test)
    print ("Energy L2err        : %e eV" % l2e)
    print ("Energy L2err/Natoms : %e eV" % l2ea)
    print ("Force  L2err        : %e eV/A" % l2f)
    print ("Virial L2err        : %e eV" % l2v)
    print ("Virial L2err/Natoms : %e eV" % l2va)

    if detail_file is not None :
        pe = np.concatenate((np.reshape(test_data["energy"][:numb_test], [-1,1]),
                             np.reshape(energy, [-1,1])), 
                            axis = 1)
        np.savetxt(os.path.join(args['system'], detail_file+".e.out"), pe, 
                   header = 'data_e pred_e')
        pf = np.concatenate((np.reshape(test_data["force"] [:numb_test], [-1,3]), 
                             np.reshape(force,  [-1,3])), 
                            axis = 1)
        np.savetxt(os.path.join(args['system'], detail_file+".f.out"), pf,
                   header = 'data_fx data_fy data_fz pred_fx pred_fy pred_fz')
        pv = np.concatenate((np.reshape(test_data["virial"][:numb_test], [-1,9]), 
                             np.reshape(virial, [-1,9])), 
                            axis = 1)
        np.savetxt(os.path.join(args['system'], detail_file+".v.out"), pv,
                   header = 'data_vxx data_vxy data_vxz data_vyx data_vyy data_vyz data_vzx data_vzy data_vzz pred_vxx pred_vxy pred_vxz pred_vyx pred_vyy pred_vyz pred_vzx pred_vzy pred_vzz')        
    return numb_test,_first_fparam(fparam),natoms, l2e, l2ea, l2f, l2v

def get_train_data(data):
    """
    data: DataSets instance
    ------
    DataSets methods can only load each set directory seperately
    This is a wrapper to concatenate all training data together
    """
    if DataSets is not None:
        train_dirs = data.train_dirs
        data.load_batch_set(train_dirs[0]);  out = data.batch_set
        
        if len(train_dirs)>1:
            for i in range(1,len(train_dirs)):
                data.load_batch_set(train_dirs[i]); add  = data.batch_set
                for key in out:
                    out[key] = np.concatenate((out[key],add[key]))
        return out

    train_sets = [data._load_set(set_dir) for set_dir in data.dirs]
    out = train_sets[0]
    for add in train_sets[1:]:
        for key in out:
            if "find_" in key:
                out[key] = out[key] or add[key]
            else:
                out[key] = np.concatenate((out[key], add[key]), axis=0)
    return out

def train_ener (inputs) :
    """
    deepmd-kit has function test_ener which deal with test_data only
    `train_ener` are for train data only
    """
    
    if inputs['rand_seed'] is not None :
        np.random.seed(inputs['rand_seed'] % (2**32))

    dp = DeepPot(inputs['model'])
    data = _new_data (inputs['system'], inputs['set_prefix'], inputs['shuffle_test'], dp=dp, set_names={"set.000"})
    _add_model_data(data, dp)
    
    train_data = get_train_data (data)
    

    numb_test = data.get_sys_numb_batch(1)  ## use 1 batch, # of batches are the numb of train
    natoms = len(train_data["type"][0])
    nframes = train_data["coord"].shape[0]
    #print("xxxxx",nframes, numb_test)
    numb_test = nframes  #, to be investigated, original dp use min, but here should be nframes directly, I think, Jan 18, 21, min(nfames, numb_test)
    coord, box, atype, fparam, aparam, extra = _load_eval_input(data, dp, train_data, numb_test)
    detail_file = inputs['detail_file']
    if detail_file is not None:
        atomic = True
    else:
        atomic = False

    ret = dp.eval(coord, box, atype, fparam = fparam, aparam = aparam, atomic = atomic, **extra)
    energy = ret[0]
    force  = ret[1]
    virial = ret[2]
    energy = energy.reshape([numb_test,1])
    force  = force.reshape([numb_test,-1])
    virial = virial.reshape([numb_test,9])
    if atomic:
        ae = ret[3]
        av = ret[4]
        ae = ae.reshape([numb_test,-1])
        av = av.reshape([numb_test,-1])

    l2e = (l2err (energy - train_data["energy"].reshape([-1,1])))
    l2f = (l2err (force  - train_data["force"]))
    l2v = (l2err (virial - train_data["virial"]))
    l2ea= l2e/natoms
    l2va= l2v/natoms

    # print ("# energies: %s" % energy)
    print ("# number of train data : %d " % numb_test)
    print ("Energy L2err        : %e eV" % l2e)
    print ("Energy L2err/Natoms : %e eV" % l2ea)
    print ("Force  L2err        : %e eV/A" % l2f)
    print ("Virial L2err        : %e eV" % l2v)
    print ("Virial L2err/Natoms : %e eV" % l2va)

    if detail_file is not None :
        pe = np.concatenate((np.reshape(train_data["energy"], [-1,1]),
                             np.reshape(energy, [-1,1])), 
                            axis = 1)
        np.savetxt(os.path.join(inputs['system'],detail_file+".e.tr.out"), pe, 
                   header = 'data_e pred_e')
        pf = np.concatenate((np.reshape(train_data["force"], [-1,3]), 
                             np.reshape(force,  [-1,3])), 
                            axis = 1)
        np.savetxt(os.path.join(inputs['system'],detail_file+".f.tr.out"), pf,
                   header = 'data_fx data_fy data_fz pred_fx pred_fy pred_fz')
        pv = np.concatenate((np.reshape(train_data["virial"], [-1,9]), 
                             np.reshape(virial, [-1,9])), 
                            axis = 1)
        np.savetxt(os.path.join(inputs['system'],detail_file+".v.tr.out"), pv,
                   header = 'data_vxx data_vxy data_vxz data_vyx data_vyy data_vyz data_vzx data_vzy data_vzz pred_vxx pred_vxy pred_vxz pred_vyx pred_vyy pred_vyz pred_vzx pred_vzy pred_vzz')        
    return numb_test,_first_fparam(fparam),natoms, l2e, l2ea, l2f, l2v

def train_force_by_type(inputs):
    """
    Evaluate training-set force L2 error for all atom types in one single dp.eval call.

    Returns
    -------
    numb_test : int
        Number of training frames.
    fparam0 : float or None
        fparam[0][0] if fparam exists, else None.
    results : list of tuple
        [(sel_type, natoms_sel, l2f_sel), ...]
        sorted by sel_type
    """

    if inputs['rand_seed'] is not None:
        np.random.seed(inputs['rand_seed'] % (2**32))

    dp = DeepPot(inputs['model'])
    data = _new_data(inputs['system'], inputs['set_prefix'],
                     inputs['shuffle_test'], dp=dp, set_names={"set.000"})
    _add_model_data(data, dp)
    train_data = get_train_data(data)

    nframes = train_data["coord"].shape[0]
    numb_test = nframes

    coord, box, atype, fparam, aparam, extra = _load_eval_input(data, dp, train_data, numb_test)
    atype_by_frame = np.asarray(atype, dtype=int)
    if atype_by_frame.ndim == 1:
        atype_by_frame = np.tile(atype_by_frame.reshape(1, -1), (numb_test, 1))

    ret = dp.eval(coord, box, atype, fparam=fparam, aparam=aparam, atomic=False, **extra)

    force_pred = ret[1].reshape([numb_test, -1, 3])
    force_ref  = train_data["force"].reshape([numb_test, -1, 3])

    unique_types = sorted(np.unique(atype_by_frame).tolist())
    results = []

    print("# number of train data     : %d" % numb_test)
    print("# force error by atom type:")

    for sel_type in unique_types:
        atom_mask = (atype_by_frame == sel_type)
        natoms_sel = int(np.sum(atom_mask[0]))

        if not np.any(atom_mask):
            continue

        force_pred_sel = force_pred[atom_mask]
        force_ref_sel  = force_ref[atom_mask]

        l2f_sel = l2err(force_pred_sel - force_ref_sel)
        results.append((int(sel_type), natoms_sel, float(l2f_sel)))

        print("#   type %d: natoms = %d, force_l2err = %e eV/A"
              % (sel_type, natoms_sel, l2f_sel))

    fparam0 = _first_fparam(fparam)
    return numb_test, fparam0, results
