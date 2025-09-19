This repository containst the files used to support a masters capstone project at ASU - LSC 586 offered in Summer B of 2025.

This work involved the use of an ensemble learning model approach to demonstrate an improvement in the the prediction of off-target cleavage in CRISPR-Cas9 experiments. Base models chosen for the ensemble were CRISPR-NET, piCRISPR, and CRISPR-Dipoff. Base model code was copied from available repositores and modified to allow for workflow in the jupyter notebook - LSC586_runBaseModels.ipynb

Source data for the training of the ensemble came from a study by José M. Uribe-Salazar et al. (2022) where an emperical dataset of off-target cleavage in zebrafish was used to evaluate the predictve accuracy of several base models. For our work we processed the NGS files provided in the zebrafish study using CIRCLE-Seq code from it's respeitive repositiory.
   Uribe-Salazar, J. M., Kaya, G., Sekar, A., Weyenberg, K., Ingamells, C., & Dennis, M. Y. (2022). Evaluation of CRISPR gene-editing tools in zebrafish. BMC genomics, 23, 1-16.

Ensemble model training and evaluatihon is provided in - addfile

For purposes of our coursework, we utilized the Sol HPC at ASU to perform this work
   Jennewein, Douglas M. et al. "The Sol Supercomputer at Arizona State University." In Practice and Experience in Advanced Research Computing (pp. 296–301). Association for Computing Machinery, 2023, doi:10.1145/3569951.3597573
