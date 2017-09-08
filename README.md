# PREDICTD
Durham T, Libbrecht M, Howbert J, Bilmes J, Noble W. PaRallel Epigenomics Data Imputation with Cloud-based Tensor Decomposition. 2017. https://doi.org/10.1101/123927.

This repository contains the code to run PREDICTD, a program to model the epigenome based on the Encyclopedia of DNA Elements and the NIH Roadmap Epigenomics Project data and to impute the results of epigenomics experiments that have not yet been done. A computing environment for running this code is distributed as an Amazon Machine Image, and it is easiest to get the code up and running by following the steps in the tutorial below to start a cluster in Amazon Web Services. This tutorial will demonstrate how to train the model on the Roadmap Consolidated data set used in the paper. The model can also be used to impute data for a new cell type, and there will be another tutorial for that use case coming soon. If you do not want to run the model, but simply want to get the imputed data from the paper, you will be able to download that data in bigwig format from the ENCODE project website soon.

##Installation

PREDICTD is most readily available on Amazon Web Services as part of an Amazon Machine Image (AMI) called PREDICTD (ami-e7b5439f).

##Instructions and Tutorials

For more examples and information about how to run PREDICTD, please see the wiki page of this GitHub repository (https://github.com/tdurham86/PREDICTD/wiki).

##License

PREDICTD is released as an open-source project under the MIT License:

MIT License

Copyright (c) 2017 tdurham86

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
