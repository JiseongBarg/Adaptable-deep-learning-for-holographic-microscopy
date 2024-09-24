# Adaptable deep learning for holographic microscopy: A case study on tissue-type and system variability in label-free histopathology


<div align=center>	
We provide pytorch(python) implementations of <b>Adaptable deep learning for holographic microscopy: A case study on tissue-type and system variability in label-free histopathology
</b>.
<br>
<br>
This code was written by <b>Jiseong Barg</b>.
<br>
<br>
Last update: 2024.09.04
</div>


# Overview

Holographic microscopy has emerged as a vital tool in biomedicine, enabling visualization of microscopic morphological features of tissues and cells in a label-free manner. Recently, deep learning (DL)-based image reconstruction models have demonstrated state-of-the-art performance in holographic image reconstruction. However, their utility in practice is still severely limited as conventional training schemes could not properly handle out-of-distribution (OOD) data. Here, we leverage back-propagation operation and reparameterization of forward propagator to enable an adaptable image reconstruction model for histopathologic inspection. Only given with a training data set of rectum tissue images captured from a single imaging configuration, our scheme consistently shows high reconstruction performance even with the input hologram of different tissue types and imaging configurations. Furthermore, we demonstrate holographic reconstruction of characteristic features of cancerous tissues while a given data set is strictly confined to normal tissues. Our results suggest that the DL-based image reconstruction approaches, with sophisticated adaptation techniques, could offer an extensively generalizable solution for inverse mapping problems in imaging.
 


 ![Overview of proposed approach](./image/main1_latest.png)

# Environment requirements

The code was tested on Windows 10 with Python and PyTorch. The required packages to reproduce the results can be found in the `requirements.txt` file. 

The following software and hardware configurations were tested and are recommended:

* Python >= 3.9
* CUDA 11.8
* cuDNN 8.4.1
* Intel i5-11400
* Nvidia RTX3080 Ti
* RAM >= 80 GB

To install the necessary packages, please use the following command:
```
pip install -r requirements.txt
```

# Test

To test the baseline and proposed methods and reproduce results depicted in figure 5,

please follow these steps:

1. **Download Pretrained Weights**
   - Access the pretrained weights from [Google Drive](https://drive.google.com/drive/folders/1iVlmMHpS6WMewwPyzc2lezP5TVCSHDiJ?usp=sharing).
   - Move the downloaded weights to the directory  `/Train/PretrainedModels`.
  
2. **Select Method and Tissue Type:**
   - Choose the method (`baseline` or `proposed`) and the tissue type (`colon` or `appendix`) for testing.

    These options correspond to the conditions used in Figure 5.

3. **Run the `main_test.py`**
   - Execute the following command in your terminal, replacing the method and tissue type as needed:
     ```
     python main_test.py --methods proposed --tissue_type appendix
     ```

4. **Reconstruction Results**
   - The reconstruction results can be found in the `/Test` directory after the script has completed running.

# Train

The data for training both methods will be available from the corresponding author upon reasonable request.

After downloading the training data, place it in the `/Datasets/train/field` directory.

1. **Select Method:**
   - Choose the method (`baseline` or `proposed`)

2. **Run the `main_train.py`**
   - Execute the following command in your terminal, replacing the method and other hyper-parameters as needed.
   
     The command below utilizes the hyper-parameters used in the paper:
     ```
     python main_train.py --methods proposed --batch_size 16 --iterations 50000 --chk_iter 500 --dist_min 1.353 --dist_max 5.413 --pix_min 0.05 --pix_max 0.25
     ```
   
3. **Training Results**
   - The results from the training process will be saved in the '/Train' directory after the script completes its execution.














