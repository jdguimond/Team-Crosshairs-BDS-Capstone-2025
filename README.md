This repository contains the files used to support a masters capstone project at ASU - LSC 586 offered in Summer B of 2025.

Notes
-----
-This work involved the use of an ensemble learning model approach to demonstrate an improvement in the the prediction of off-target cleavage in CRISPR-Cas9 experiments. Base models chosen for the ensemble were CRISPR-NET, piCRISPR, and CRISPR-Dipoff. Base model code was copied from published repositories and modified to allow for the processing workflow in the jupyter notebook - LSC586_runBaseModels.ipynb

https://github.com/JasonLinjc/CRISPR-Net

https://github.com/florianst/picrispr

https://github.com/tzpranto/CRISPR-DIPOFF

-Source data for the training of the ensemble was taken from a study by José M. Uribe-Salazar et al. (2022) where an empirical dataset of off-target cleavage in zebrafish was used to evaluate the predictive accuracy of several base models. For our work we processed the NGS files provided in this study using CIRCLE-seq processing pipeline.

https://github.com/tsailabSJ/circleseq

-Ensemble model training and evaluation is provided in - Ensemble Script 1a.ipynb. At the moment the output of piCRISPR requires manual merging with the piCRISPR input file, as well as, the concatenation of CRISPR-DIPOFF output files based on the need to limit CRISPR-DIPOFF input files to sequences of the same length.

-For purposes related to our coursework, we utilized the Sol HPC at ASU to perform this work, although not necessary
   

References
-----
Uribe-Salazar, J. M., Kaya, G., Sekar, A., Weyenberg, K., Ingamells, C., & Dennis, M. Y. (2022). Evaluation of CRISPR gene-editing tools in zebrafish. BMC genomics, 23, 1-16

Jennewein, Douglas M. et al. "The Sol Supercomputer at Arizona State University." In Practice and Experience in Advanced Research Computing (pp. 296–301). Association for Computing Machinery, 2023, doi:10.1145/3569951.3597573
