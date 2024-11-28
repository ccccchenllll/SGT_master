import random
from data import ImageDetectionsField,MatrixField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer.transformer import Transformer
from models.transformer import MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch

from tqdm import tqdm

import torch.nn as nn

import argparse, os, pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import warnings
warnings.filterwarnings("ignore")
import os, json
from torch.utils.tensorboard import SummaryWriter
# from Evison import Display, show_network
import matplotlib.pyplot as plt
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='evaluation', unit='it', total=len(dataloader)) as pbar:

        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            bs = images.size(0)

            ad_matrix = caps_gt[0]
            ad_matrix = torch.cat(ad_matrix, dim=1).reshape(bs, 10, 10).to(device)
            # print(ad_matrix)
            # print(ad_matrix.type)

            caps_gt = caps_gt[1]
            # print(caps_gt)

            # images = images

            with torch.no_grad():
                out, _ = model.beam_search(images, ad_matrix, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            print(caps_gen)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()
            
    if not os.path.exists('predict_caption'):
        os.makedirs('predict_caption')
    json.dump(gen, open('predict_caption/predict_caption2.json', 'w'))
    json.dump(gts,open('predict_caption/original_caption2.json', 'w'))

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--m', type=int, default=48)
    parser.add_argument('--features_path', default='/data/zfzhu/lc/m2transformer/features/instruments18_caption/')   #特征
    #parser.add_argument('--features_path', default='/data/zfzhu/lc/m2transformer/features/instruments18_caption/')   #特征
    parser.add_argument('--annotation_folder', type=str, default = '/data/zfzhu/lc/m2transformer/annotations/annotations')   #标记
    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=10, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    ad_matrix_field = MatrixField(detections_path=args.features_path, max_detections=10,load_in_tmp=False)



    # Create the dataset


    dataset = COCO(image_field,ad_matrix_field,text_field, args.features_path, args.annotation_folder, args.annotation_folder)

    train_dataset, val_dataset = dataset.splits

    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    memory = np.load('memory48.npy')
    memory = memory[np.newaxis,:]

    # Model and dataloaders

    encoder = MemoryAugmentedEncoder(3, 0, memory, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])

    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load('/data/zfzhu/lc/SGT-MICCAI/saved_models/m2_transformer_best.pth')
    #print(model)




    model.load_state_dict(data['state_dict'])
    print("Epoch %d" % data['epoch'])
    print(data['best_cider'])



    dict_dataset_val = val_dataset.image_dictionary({'image': image_field,'ad_matrix':ad_matrix_field, 'text': RawField()})

    dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)

    scores = evaluate_metrics(model, dict_dataloader_val, text_field)
    print(scores)