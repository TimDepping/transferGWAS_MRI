#!/bin/bash

alias plink2=/dhc/projects/mpws2022cl1/plink2

INDIV=indiv.txt



# very soft filtering parameters:
# minor allele frequency
MAF=0.001

# Hardy-Weinberg p-value
HWE_PVAL=0.001

# LD-pruning, R^2, window-size & number of steps (see plink2 documentation for details)
LD_THRESHOLD=0.8
LD_WINDOW="500 kb"
LD_STEP=1



TMP_DIR=tmp_dir_del/
mkdir ${TMP_DIR}

# for i in {1..22}; do
for i in {1..3}; do
    INPUT="/dhc/projects/ukbiobank/original/genetics/microarray/unzipped/ukb_chr${i}_v2"
	OUTPUT="${PWD}/lmm/preprocessing_ma_output/chr${i}"

    TMPFILE=${TMP_DIR}/chr${i}.tmp
	LD_TMP=${TMP_DIR}/ld
	# individuals, Hardy-Weinberg, and MAF 
	plink2 --bfile ${INPUT} --keep ${INDIV} --hwe ${HWE_PVAL} --maf ${MAF} --make-bed --out ${TMPFILE}
	# LD pruning
	plink2 --bfile ${TMPFILE} --indep-pairwise ${LD_WINDOW} ${LD_STEP} ${LD_THRESHOLD} --out ${LD_TMP}
	plink2 --bfile ${TMPFILE} --extract ${LD_TMP}.prune.in --make-bed --out ${OUTPUT}
done

rm -r ${TMP_DIR}
