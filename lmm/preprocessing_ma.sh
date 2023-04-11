#!/bin/bash

alias plink2=/dhc/projects/mpws2022cl1/plink2

OUTPUT_DIR=$1
INDIV=$2
TMP_DIR=$OUTPUT_DIR/.tmp

# very soft filtering parameters:
# minor allele frequency
MAF=0.001

# Hardy-Weinberg p-value
HWE_PVAL=0.001

# LD-pruning, R^2, window-size & number of steps (see plink2 documentation for details)
LD_THRESHOLD=0.8 # paires greater this value are pruned 
LD_WINDOW="500 kb"
LD_STEP=1

rm -r ${TMP_DIR}
rm -r ${OUTPUT_DIR}
mkdir -p ${TMP_DIR}
mkdir -p ${OUTPUT_DIR}

for i in {1..22}; do
    INPUT="/dhc/projects/ukbiobank/original/genetics/microarray/unzipped/ukb_chr${i}_v2"
	OUTPUT="${OUTPUT_DIR}/chr${i}"

    TMPFILE=${TMP_DIR}/chr${i}.tmp
	LD_TMP=${TMP_DIR}/ld
	# individuals, Hardy-Weinberg, and MAF 
		# bfile: Define input reference (automatically combined .bed/.bim/.fam)
		# keep: remove all unlisted samples 
		# hwe: filters out all variants which have Hardy-Weinberg equilibrium exact test p-value below the provided threshold 
		# maf: filters out all variants with allele frequency below the provided threshold
		# make-bed: creates a .bed file
		# out: PREFIX of output file (combined with make-bed = PREFIX.bed)

	# Filter individuals by Hardy-Weinberg equilibrium (hwe) tests and min allele frequencies (maf)
	# --make-bed: save the data in PLINK 1 binary format
	plink2 --bfile ${INPUT} --keep ${INDIV} --hwe ${HWE_PVAL} --maf ${MAF} --make-bed --out ${TMPFILE}
	# Linkage disequilibrium (LD) pruning

	# LD pruning
		# input = filtered plink files
		# indep-pairwise: LD pruning 
		# bad-ld = cannot estimate LD effectively when very few founders are present -> skip this error with this argument
		# output: pruned subset of variants that are in approximate linkage equilibrium with each other, 
		# writing the IDs to plink2.prune.in
		# and the IDs of all excluded variants (greater threshold > 0.8) to plink2.prune.out.
	plink2 --bfile ${TMPFILE} --indep-pairwise ${LD_WINDOW} ${LD_STEP} ${LD_THRESHOLD} --out ${LD_TMP} --bad-ld
	
	# extract: removes all unlisted variants from the current analysis.
	plink2 --bfile ${TMPFILE} --extract ${LD_TMP}.prune.in --make-bed --out ${OUTPUT}
done

rm -r ${TMP_DIR}
