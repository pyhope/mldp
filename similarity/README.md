# Similarity

Post-processing scripts for the soap similiarity analysis

```
Step 1: prepare extended xyz file (from dump file)
        For mass density, use dump2xyz.py
        For density based on k value, e.g., similiarity metric, one needs to run LAMMPS PLUMED simulations and use merge_xyz.py
Step 2: calculate the density vs. planar surface or proximity
Step 3: fit density to Gibbs Dividing surface forumula
Step 4: Count number of atoms in each phase
        planar: sum_counts*
        proximity: sum_proximity*
```

```
mkdir soap
cp in.lammps_360pv108w /u/scratch/j/jd848/360pv108w_75g/2500/soap
lmp -in in.lammps_360pv108w  # better sumbit job
python ~/script/mldp/similarity/merge_xyz.py
python ~/script/mldp/similarity/bash_sub.py -nw 2 -m mass -p 2 -i 70
```



`merge_xyz.py` merges the soap similarity analyzed file of each element into one .xyz file => no speicial dependencies



`stat2_pv_20g.py` presents an example of using Gibbs dividing surface to analyze the interface, routines in the `stat_lib.py` are called.  => speicial dependencies : MDAanalysis, lmfit, ase 

`stat_lib_mass.py`  collection of scripts, mass as the weighting factor





==TODO==

1- merge stat_lib.py and stat_lib_mass.py =>done

weight can be specify by the user

2- any amounts elements  => done

3- flexibility of modulate the 

4-more organized ch output

l1 l2 l3 l4 .. s1 s2 s3 s4 ... int1 int2 s3 s4

Testing folder: /Users/jiedeng/GD/papers/zhixue/exsolution/w/md/mgofew/256mgo4096fe16w/4.5k/cont1/soap


mass as weight 

bash_sub_mass.py

stat_lib_mass.py

stat_mass.py
