#!/bin/bash

alias plink2=/dhc/projects/mpws2022cl1/plink2

OUTPUT_DIR=$1
INDIV=$2
# number of threads to use
THREADS=$3

TMP_DIR=$OUTPUT_DIR/.tmp

# Hardy-Weinberg p-value (1e-12)
HWE_PVAL=0.000000000001

# minor allele frequency
MAF=0.001

rm -r ${TMP_DIR}
rm -r ${OUTPUT_DIR}
mkdir -p ${TMP_DIR}
mkdir -p ${OUTPUT_DIR}

for i in {1..22}; do
	INPUT="/dhc/projects/ukbiobank/original/genetics/imputed/EGAD00010001474/ukb_imp_chr${i}_v3.bgen"
	SAMPLE="/dhc/projects/ukbiobank/original/genetics/imputed/sample/ukb40502_imp_chr${i}_v3_s487296.sample"
	OUTPUT="${OUTPUT_DIR}/chr${i}"

	plink2 --bgen ${INPUT} 'ref-first' --sample ${SAMPLE} --keep ${INDIV} --maf ${MAF} --hwe ${HWE_PVAL} --threads ${THREADS} --export bgen-1.2 bits=8 --out ${OUTPUT}

done
