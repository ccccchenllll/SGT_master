# SGT_master
Prerequisites 
 - tqdm, h5py, spacy, etc
 - python 3.8
 - pytorch (1.4, or higher version)
 
Data preparation:
We preprocess the data given by https://github.com/XuMengyaAmy/ReportDALS. If you need processed data, please contact me by email.

Evaluation:
To reproduce the results reported in our paper, download the pretrained model file https://drive.google.com/file/d/1OnzMWnct939E0RB_6nMVstVTOcKei2MM/view?usp=sharing and place it in the code folder.

Run python val_base.py. Note that please change the file path.

Training procedure:
Run python train_base.py

