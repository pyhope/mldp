#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 00:21:59 2020
Step PotEng KinEng TotEng Temp Press Volume 
@author: jiedeng
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def blockAverage(datastream, isplot=False, maxBlockSize=0):
	"""This program computes the block average of a potentially correlated timeseries "x", and
	provides error bounds for the estimated mean <x>.
	As input provide a vector or timeseries "x", and the largest block size.

	Check out writeup in the following blog posts for more:
	http://sachinashanbhag.blogspot.com/2013/08/block-averaging-estimating-uncertainty_14.html
	http://sachinashanbhag.blogspot.com/2013/08/block-averaging-estimating-uncertainty.html
	"""

	Nobs         = len(datastream)           # total number of observations in datastream
	minBlockSize = 1;                        # min: 1 observation/block

	if maxBlockSize == 0:
		maxBlockSize = int(Nobs/4);        # max: 4 blocs (otherwise can't calc variance)

	NumBlocks = maxBlockSize - minBlockSize   # total number of block sizes

	blockMean = np.zeros(NumBlocks)               # mean (expect to be "nearly" constant)
	blockVar  = np.zeros(NumBlocks)               # variance associated with each blockSize
	blockCtr  = 0

				#
				#  blockSize is # observations/block
				#  run them through all the possibilities
				#

	for blockSize in range(minBlockSize, maxBlockSize):

		Nblock    = int(Nobs/blockSize)               # total number of such blocks in datastream
		obsProp   = np.zeros(Nblock)                  # container for parcelling block

		# Loop to chop datastream into blocks
		# and take average
		for i in range(1,Nblock+1):

			ibeg = (i-1) * blockSize
			iend =  ibeg + blockSize
			obsProp[i-1] = np.mean(datastream[ibeg:iend])

		blockMean[blockCtr] = np.mean(obsProp)
		blockVar[blockCtr]  = np.var(obsProp)/(Nblock - 1)
		blockCtr += 1

	v = np.arange(minBlockSize,maxBlockSize) # detailed info

	if isplot:

		plt.subplot(2,1,1)
		plt.plot(v, np.sqrt(blockVar),'ro-',lw=2)
		plt.xlabel('block size')
		plt.ylabel('std')

		plt.subplot(2,1,2)
		plt.errorbar(v, blockMean, np.sqrt(blockVar))
		plt.ylabel('<x>')
		plt.xlabel('block size')

#		print "<x> = {0:f} +/- {1:f}\n".format(blockMean[-1], np.sqrt(blockVar[-1]))

		plt.tight_layout()
		plt.show()

	return blockMean[-1], np.sqrt(blockVar)[-1]




def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order
    read big file in reverse order is not trivial, library avaialable but not a optimum choice
    source: https://stackoverflow.com/questions/2301789/read-a-file-in-reverse-order-using-python
    """

    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first 
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment
            
def search_header(file,pattern):
    fp = open(file)
    for i in range(int(1000000)):
        line = next(fp)
        if pattern in line:
            return i,line
    
def search_footer(file,pattern):
    txt  = reverse_readline(file)
    for i in range(int(100000)):
        line = next(txt)
        if pattern in line:
            return i

def autocorr(a):
    b=np.concatenate((a,np.zeros(len(a))),axis=0)
    c= np.fft.ifft(np.fft.fft(b)*np.conjugate(np.fft.fft(b))).real
    d=c[:len(c)//2]
    d=d/(np.array(range(len(a)))+1)[::-1]
    return d

#print(skip_header[1])
#if 'Pxx Pyy Pzz Pxy Pyz Pxz' in skip_header[1]:
#    from scipy.integrate import cumtrapz
#    stress = out[args.skip_step:,7:]*1e5
##    print(stress[:,0])
#    c_xx = autocorr(stress[:,0])
#    c_xy = autocorr(stress[:,3])
#    c_xz = autocorr(stress[:,5])
#    c_yy = autocorr(stress[:,1])
#    c_yz = autocorr(stress[:,4])
#    c_zz = autocorr(stress[:,2])
#
#    corr = (c_xy + c_xz + c_xy)/3
#    corr_2 = (c_xy + c_xz + c_xy + autocorr((stress[:,0] - stress[:,1])/2) + autocorr((stress[:,1] - stress[:,2])/2))/5
#    
#    fs2s       = 1e-15
##    eV_A3_2_Pa = 160.21766208e9
#    kb         = 1.38e-23
#    vol  = out[:,6][0]
#    temp = np.mean(out[:,4])
#    visco      = cumtrapz(corr,list(range(len(corr))))*fs2s*vol*1e-30/(kb*temp*3)
#    visco_2    = cumtrapz(corr_2,list(range(len(corr))))*fs2s*vol*1e-30/(kb*temp*3)
#    
#    plt.figure()
#    plt.plot(list(range(len(visco))), visco,label='pii')
#    plt.plot(list(range(len(visco_2))), visco_2,label='pii+(pii-pjj)')
#    plt.yscale('log')
#    plt.legend()
#    plt.savefig("visco.png",bbox_inches='tight')
