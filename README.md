# SSGDL
Pytorch implementation for codes in "Scale-Selectable Global Information and Discrepancy Learning Network for Multimodal Sentiment Analysis"(https://doi.org/10.1109/TAFFC.2025.3580779)
<img width="15589" height="6625" alt="Overall Model" src="https://github.com/user-attachments/assets/d9c81c9e-193c-45f1-aa4e-c7d29a790ed2" />
# Prepare
Download the MOSI and MOSEI pkl file (https://drive.google.com/drive/folders/1_u1Vt0_4g0RLoQbdslBwAdMslEdW1avI?usp=sharing). Put it under the "./dataset" directory.

Download the SentiLARE language model files (https://drive.google.com/file/d/1onz0ds0CchBRFcSc_AkTLH_AZX_iNTjO/view?usp=share_link), and then put them into the "./pretrained-model/sentilare_model" directory.

# Run
```python
python train.py
```



Note: the scale of MOSI dataset is small, so the training process is not stable. To get results close to those in SSGDL paper, you can set the seed in args to 6758. The experimental results of this paper are obtained on the Windows system.

# Paper
```
@ARTICLE{11039712,
  author={He, Xiaojiang and Pan, Yushan and Guo, Xinfei and Xu, Zhijie and Yang, Chenguang},
  journal={IEEE Transactions on Affective Computing}, 
  title={Scale-Selectable Global Information and Discrepancy Learning Network for Multimodal Sentiment Analysis}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Sentiment analysis;Data mining;Correlation;Visualization;Transformers;Affective computing;Main-secondary;Feature extraction;Depression;Training;Multimodal Sentiment Analysis;depression detection;Scale-Selectabl Global Information;Inter-modal Discrepancy Learning;Neuro-scientific theories},
  doi={10.1109/TAFFC.2025.3580779}}
```

