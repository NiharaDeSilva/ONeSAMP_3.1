#!/usr/bin/python
import sys

import math
import os
import re

import numpy as np
from numba import njit, prange

import collections
from decimal import *

m = {'0101': [0, 0],
     '0102': [0, 1],
     '0103': [0, 2],
     '0104': [0, 3],
     '0201': [1, 0],
     '0202': [1, 1],
     '0203': [1, 2],
     '0204': [1, 3],
     '0301': [2, 0],
     '0302': [2, 1],
     '0303': [2, 2],
     '0304': [2, 3],
     '0401': [3, 0],
     '0402': [3, 1],
     '0403': [3, 2],
     '0404': [3, 3],
     '0000': [-1,-1],
     '0100': [0,-1],
     '0001': [-1,0],
     '0200': [1,-1],
     '0300': [2,-1],
     '0400': [3,-1],
     '0002': [-1,1],
     '0003': [-1,2],
     '0004': [-1,3]}

elements = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])


@njit(fastmath=False, parallel=True)
def compute_stat1(data):
    sample_size, num_loci, _ = data.shape
    total_spots = np.float64(sample_size * 2)
    delete_cols = []

    elements = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    allcnt = []
    homoloci_array = []
    di = []

    for i in range(num_loci):
        temp = data[:, i, :]
        ref_allele = temp[0, 0]

        homo_match = 0
        for k in range(sample_size):
            if temp[k, 0] == ref_allele and temp[k, 1] == ref_allele:
                homo_match += 1
        homoloci = np.float64(homo_match) / np.float64(sample_size)

        # allHomoloci computation
        matches = np.zeros(elements.shape[0])
        for e_idx in range(elements.shape[0]):
            count = 0
            for j in range(sample_size):
                if temp[j, 0] == elements[e_idx, 0] and temp[j, 1] == elements[e_idx, 1]:
                    count += 1
            matches[e_idx] = count
        all_homo_count = np.sum(matches)
        allHomoloci = np.float64(all_homo_count) / np.float64(sample_size)
        homoloci_array.append(allHomoloci)

        curr_cnt = 0
        for j in range(sample_size):
            if temp[j, 0] == ref_allele:
                curr_cnt += 1
            if temp[j, 1] == ref_allele:
                curr_cnt += 1
        allcnt.append(curr_cnt)

        if curr_cnt == total_spots:
            delete_cols.append(i)

        freq_sq = (np.float64(curr_cnt) / total_spots) ** 2
        di.append(homoloci - freq_sq)

    # filter polymorphic loci
    keep_indices = []
    for i in range(num_loci):
        if i not in delete_cols:
            keep_indices.append(i)

    num_loci = len(keep_indices)
    if num_loci < 2:
        return np.float64(0.0), np.array(allcnt), np.array(homoloci_array)

    samp_correction = np.float64(2.0) / (np.float64(num_loci) * (np.float64(num_loci) - 1))
    data_filtered = np.empty((sample_size, num_loci, 2), dtype=np.int64)

    new_allcnt = []
    new_di = []
    new_homoloci = []

    for idx, i in enumerate(keep_indices):
        data_filtered[:, idx, :] = data[:, i, :]
        new_allcnt.append(allcnt[i])
        new_di.append(di[i])
        new_homoloci.append(homoloci_array[i])

    allcnt = new_allcnt
    di = new_di
    homoloci_array = new_homoloci

    r = np.float64(0.0)

    for i in range(num_loci):
        if di[i] == 0:
            continue
        lociA = data_filtered[:, i, :]
        refA = lociA[0, 0]
        index_A = np.zeros(sample_size, dtype=np.int64)
        for k in range(sample_size):
            if lociA[k, 0] == refA and lociA[k, 1] == refA:
                index_A[k] = 1

        for j in range(i + 1, num_loci):
            if di[j] == 0:
                continue
            lociB = data_filtered[:, j, :]
            refB = lociB[0, 0]
            index_B = np.zeros(sample_size, dtype=np.int64)
            for k in range(sample_size):
                if lociB[k, 0] == refB and lociB[k, 1] == refB:
                    index_B[k] = 1

            hits = np.sum(index_A * index_B)

            curr_cntA = 0
            for k in range(sample_size):
                if lociA[k, 0] == refA:
                    curr_cntA += 1
                if lociA[k, 1] == refA:
                    curr_cntA += 1

            curr_cntB = 0
            for k in range(sample_size):
                if lociB[k, 0] == refB:
                    curr_cntB += 1
                if lociB[k, 1] == refB:
                    curr_cntB += 1

            ai = np.float64(curr_cntA) / total_spots
            bj = np.float64(curr_cntB) / total_spots

            denom = (ai * (1.0 - ai) + di[i]) * (bj * (1.0 - bj) + di[j])
            if denom == 0:
                continue

            jointAB = np.float64(hits) / (np.float64(2.0) * total_spots)
            r += ((jointAB - ai * bj) ** 2) / denom

    stat1 = r * samp_correction
    return stat1, np.array(allcnt), np.array(homoloci_array)


class statisticsClass:
    ####### Members of Class

    ARRAY_MISSINGIndiv = []
    ARRAY_MISSINGLoci = []
    data = []  ## Matrix is [individuals by loci]
    stat1 = 0
    stat1_new = 0
    stat2 = 0
    stat3 = 0
    stat4 = 0
    stat5 = 0
    numLoci = 0
    sampleSize = 0  ##Indivduals
    NE_VALUE = 0
    DEBUG = 0
    result = []
    allcnt = []
    homoLoci = []
    homoLociCol = []
    hexp = []
    hob = []

    # m is the transfer array

    ######################################################################
    # writeStatistics                                                   ##
    ######################################################################
    # def writeStatistics(self, myFileName):
    # outputFile = open(myFileName, "w")

    ######################################################################
    # readData                                                          ##
    ######################################################################
    def readData(self, myFileName):
        with open(myFileName, 'r') as f:
            lines = f.readlines()
        result = []
        for line in lines[1:]:
            if (len(line.split()) > 10):
                result.append([m[i] for i in line.split()[2:]])
        data = np.asarray(result)
        self.data = data
        last_line = self.get_file_last_line(os.path.abspath(myFileName))
        test = last_line.strip().split(" ")
        NE_VALUEtemp = 0
        if len(test) != self.data.shape[1] + 2:
            NE_VALUEtemp = int(float(test[0]))
        self.NE_VALUE = NE_VALUEtemp
        self.numLoci = self.data.shape[1]
        if self.numLoci > 5000:
            print("error: loci size should be smaller than 5000")
            sys.exit(1)
        self.sampleSize = self.data.shape[0]


    def filterMonomorphicLoci(self):
        data = self.data
        deleteCol = []
        for i in range(self.numLoci):
            temp = data[:,i,:]
            if len(np.unique(temp)) == 1:
                deleteCol.append(i)
        print(deleteCol)
        if len(deleteCol) != 0:
            newData = np.delete(data, deleteCol, axis = 1)
            self.data = newData
            self.numLoci = self.numLoci - len(deleteCol)
            print("filter monomorphic loci")


    def get_file_last_line(self, inputfile):
        with open(inputfile, 'rb') as f:
            first_line = f.readline()
            offset = -50
            while True:
                f.seek(offset, 2)
                lines = f.readlines()
                if len(lines) >= 2:
                    last_line = lines[-1]
                    break
                offset *= 2

            return last_line.decode()

    ######################################################################
    # filterIndividuals                                                 ##
    # Filters the data for all individuals that have > 20% missing data ##
    ######################################################################
    def filterIndividuals(self, PERCENT_MISSINGIndiv):
        data = self.data
        deleteRow = []
        # to each row, if there exist 0*00 delete this row.
        for i in range(self.sampleSize):
            numMissing = 0
            temp = data[i,:,:]
            numMissing += np.sum(np.logical_or(temp[:,0] == -1, temp[:,1] == -1)) - np.sum((temp==[-1,-1]).all())
            if numMissing > self.numLoci * PERCENT_MISSINGIndiv:
                deleteRow.append(i)
        if len(deleteRow) != 0:
            newData = np.delete(data, deleteRow, axis = 0)
            self.data = newData
            self.sampleSize = self.sampleSize - len(deleteRow)




    ######################################################################
    # filterLoci                                                        ##
    ######################################################################
    def filterLoci(self, PERCENT_MISSINGLoci):
        data = self.data
        deleteCol = []

        for i in range(self.numLoci):
            temp = data[:,i,:]
            numMissing = 0
            numMissing += np.sum(np.logical_or(temp[:,0] == -1, temp[:,1] == -1)) - np.sum((temp==[-1,-1]).all())

            if numMissing > self.sampleSize * PERCENT_MISSINGLoci:
                deleteCol.append(i)

        if len(deleteCol) != 0:
            newData = np.delete(data, deleteCol, axis = 1)
            self.data = newData
            self.numLoci = self.numLoci - len(deleteCol)


    ######################################################################
    # stat1 BW Estimator                                                ##
    ######################################################################
    # TODO: when we have small individuals the r result is slightly different from the LDNE
    def test_stat1(self):
        if self.DEBUG:
            print("printing for stat1 begin: ")

        data = self.data

        numloci = data.shape[1]
        sampleSize = data.shape[0]
        # compute the frequency of each allele
        allcnt = []
        # compute the homoloci number for data matrix
        homolociArray = []
        di = []
        totalspots = sampleSize * 2
        running_sum = 0
        sampCorrection = 2 / (numloci * (numloci - 1))
        r = 0
        index = 0
        deletecol=[]
        # can be optimized. Merge it into the next for loop
        for i in range(numloci):
            temp = data[:, i, :]
            # Can be optimized
            # test = (temp == [0,0] + temp == [1,1] + temp == [2,2] + temp == [3,3]).any()

            homoloci = np.sum(np.logical_and(temp[:, 1] == temp[:, 0], temp[:, 1] == temp[0][0],
                                             temp[:, 0] == temp[0][0]) == True) / sampleSize

            allHomoloci = np.sum(np.count_nonzero((temp[:, np.newaxis, :] == elements).all(axis=2), axis=0)) / sampleSize
            homolociArray.append(allHomoloci)
            currCnt = np.sum(temp == temp[0][0])
            if currCnt == totalspots:
                deletecol.append(i)
            allcnt.append(currCnt)
            currDi = homoloci - ((currCnt / totalspots) ** 2)
            di.append(currDi)


# delete the poly column and recalculate the numloci and sample size
        if len(deletecol) != 0:
            data = np.delete(data, deletecol, axis=1)
            numloci = data.shape[1]
            sampleSize = data.shape[0]
            new_di = [di[i] for i in range(len(di)) if i not in deletecol]
            di = new_di
            sampCorrection = 2 / (numloci * (numloci - 1))
        for i in range(numloci):
            if di[i] == 0:
                continue
            LociA = data[:,i,:]
            for j in range(i + 1, numloci):
                if di[j] == 0:
                    continue
                LociB = data[:, j, :]
                index_A = np.sum((LociA == LociA[0][0]).astype(int),axis=1)
                index_B = np.sum((LociB == LociB[0][0]).astype(int),axis=1)
                hits = np.sum(index_A*index_B)
                currCntA = np.sum(LociA == LociA[0][0])
                currCntB = np.sum(LociB == LociB[0][0])
                ai = float(currCntA / totalspots)
                bj = float(currCntB / totalspots)

                if ai * (1 - ai) + di[i] == 0 or bj * (1 - bj) + di[j] == 0:
                    continue
                jointAB = float(hits / (2 * totalspots))

                denominator = (ai * (1 - ai) + di[i]) * (bj * (1 - bj) + di[j])

                r_intermdediate = float((jointAB - ai*bj) ** 2) / denominator

                r += r_intermdediate


        running_sum = r
        self.allcnt = allcnt
        self.stat1 = running_sum * sampCorrection
        self.homoLociCol = homolociArray


        if (self.DEBUG):
            print("printing for teststat1 end   ---->", self.stat1)


    ######################################################################
    # stat1 - New Stat calculation                     ##
    ######################################################################
    def test_stat1_new(self):
        if self.DEBUG:
            print("Running stat1 with shape:", self.data.shape)
        stat1, allcnt, homoLociCol = compute_stat1(self.data)
        self.stat1_new = stat1
        self.allcnt = allcnt
        self.homoLociCol = homoLociCol


    ######################################################################
    # stat2 First Moment of Multilocus Homozygosity                     ##
    ######################################################################

    def test_stat2(self):
        data = self.data
        homolociRow = []
        for i in range(self.sampleSize):
            temp = data[i,:,:]
            homoloci = np.sum(temp[:, 1] == temp[:, 0])
            homolociRow.append(homoloci)

        self.homoLoci = homolociRow
        self.stat2 = np.mean(homolociRow)

        if (self.DEBUG):
            print("(First moment of homozygosity) test Stats2 is ", self.stat2)



    ######################################################################
    # stat3 Second Moment of Multilocus Homozygosity                    ##
    ######################################################################

    def test_stat3(self):
        homolociRow = self.homoLoci
        self.stat3 = np.var(homolociRow, ddof=1)

        if self.DEBUG:
            print("(Second moment of multilocus homozygosity) Stats3 is ", self.stat3)


    ######################################################################
    # stat4 Updated after meeting w Dav                                 ##
    ######################################################################

    def test_stat4(self):

        # According to the gene diversities equation,
        # if the observe is 0, then the expected will also be 0

        data = self.data
        totalNum = self.sampleSize * 2
        hexp = np.asarray(self.hexp)
        heob = np.asarray(self.hob)
        # hexp = hexp[hexp != 0]
        # heob = heob[heob != 0]
        # fis = 1 - heob/hexp

        fis = np.zeros_like(hexp)
        mask = hexp != 0
        fis[mask] = 1 - heob[mask] / hexp[mask]
        fis[np.logical_and(hexp == 0, heob == 0)] = 0
        if mask.size > 0 and heob.size > 0 and not mask[0] and heob[0] == 0:
            fis[0] = 0

        self.stat4 = np.sum(fis) / self.numLoci


        if (self.DEBUG):
            print("New stat4:   ", newstat4)

        ######################################################################
        # stat5 Expected Heterozygosity                                     ##
        ######################################################################
        # slightly different form paper. Need to be reviewed.

    #Currently, we have the missing data not filted.
    # thus, slightly different from the David example

    def test_stat5(self):
        sampleCorrection = self.sampleSize / (self.sampleSize - 1)
        totalNum = self.sampleSize * 2
        homoloci = np.asarray(self.homoLociCol)
        data = self.data
        allcnt = np.asarray(self.allcnt)
        freqA = (allcnt / totalNum)
        freqB = 1 - freqA
        h_obser = (1-homoloci)/totalNum
        freqRes = (1 - freqA ** 2 - freqB ** 2 - h_obser)*sampleCorrection
        self.hexp = freqRes
        self.hob = h_obser*totalNum
        self.stat5 = np.sum(freqRes) / self.numLoci

        if (self.DEBUG):
            print("(Expected heterozygosity) stat5 is ", self.stat5)


