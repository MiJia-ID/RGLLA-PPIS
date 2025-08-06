# USPDB: A Novel U-Shaped Equivariant Graph Neural Network with Subgraph Sampling for Protein-DNA Binding Site Prediction


## Abstract
Accurate prediction of protein-protein interaction sites(PPIS)  plays a crucial role in understanding protein function, elucidating disease mechanisms, and facilitating drug target discovery. Although conventional approaches based on sequence or structural features have shown promising results, they still face challenges such as oversmoothing in deep graph neural networks and poor generalization to domain-specific data. To address these issues, we propose RGLLA-PPIS, a novel multimodal prediction model that integrates retrieval-augmented learning and residual graph neural networks for PPIS identification. In RGLLA-PPIS, protein graphs are constructed by combining AlphaFold3-predicted protein structures with multiple sequence-derived features. To effectively capture both local and global spatial dependencies, the model employs EGNN and GCN modules with residual connections, which help alleviate the oversmoothing problem and preserve node-level variability. Moreover, during the prediction, we used the retrieval-augmented knowledge provided by the pre-trained protein language model Evolla and ChatGPT-4o to construct semantic priors, in order to supplement potential functional site information and enhance the generalization capacity of the prediction model. Extensive experiments on benchmark datasets show that RGLLA-PPIS outperforms several state-of-the-art baselines in both accuracy and robustness. Furthermore, comparison with wet-lab results on a domain-specific protein system reveals that experimentally validated functional sites fall within the high-probability regions predicted by RGLLA-PPIS, highlighting its potential to guide real-world protein engineering tasks. 

<div align=center>
<img src="RGLLA-PPIS.jpg" width=75%>
</div>


## Preparation
### Environment Setup
```python 
   git clone https://github.com/MiJia-ID/RGLLA-PPIS.git
   conda env create -f rglla_ppis_environment.yml
```
You also need to install the relative packages to run ESM2, ProtTrans and ESM3 protein language model. 

## Experimental Procedure
### Create Dataset
**Firstly**, you need to use AlphaFold3 to obtain the CIF files for proteins. Next, you should run the cif_to_pdb.py, located in the extract_feature folder, to convert the CIF files into PDB files. Protein files are found in the training and test datasets (train_186_164, Test_315, and test_71). For more details, please refer to: https://deepmind.google/science/alphafold/alphafold-server.

Then, run the script below to create node features (PSSM, SS, AF, One-hot encoding). The file is located in the scripts folder.
```python 
python3 data_io.py 
```

**Secondly** , run the script below to create node features(ESM2 embeddings and ProtTrans embeddings). The file can be found in feature folder.</br>

```python 
python3 ESM2_5120.py 
```
```python 
python3 ProtTrans.py 
```
We choose the esm2_t48_15B_UR50D() pre-trained model of ESM-2 which has the most parameters. More details about it can be found at: https://huggingface.co/facebook/esm2_t48_15B_UR50D   </br>
We also choose the prot_t5_xl_uniref50 pre-trained model of ProtTrans, which uses a masked language modeling(MLM). More details about it can be found at: https://huggingface.co/Rostlab/prot_t5_xl_uniref50    </br>

**Thirdly**, run the script below to create edge features. The file can be found in feature folder.
```python 
python3 create_edge.py 
```

### Model Training
Run the following script to train the model.
```python
python3 train_val_bestAUPR_predicted.py 
```
**We also provide pre-trained models at** https://drive.google.com/drive/my-drive  </br>

### Inference on Pretrained Model
Run the following script to test the model. Both test datasets, DNA_129_Test and DNA_181_Test , were included in the testing of the model.
```python
python3 test_129_final.py 
```
```python
python3 test_181_final.py 
```

Additionally, During the testing phase, you can connect to the Evolla API(http://www.chat-protein.com/). Use the prompt to retrieve augmented responses and then leverage ChatGPT-4o(https://chatgpt.com/) to extract potential binding site information from these responses. Finally, combine the extracted site information with the model's predicted results to generate the final prediction.

