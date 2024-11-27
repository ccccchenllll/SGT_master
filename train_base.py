import random
from data import ImageDetectionsField,MatrixField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer.transformer import Transformer
from models.transformer import MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
import torch.nn.functional as F
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

# lines below to make the training reproducible
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



os.environ["CUDA_VISIBLE_DEVICES"] = "9"

torch.cuda.empty_cache()


#测试损失
def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, (detections,ad_matrix, captions) in enumerate(dataloader):
                detections, ad_matrix, captions = detections.to(device), ad_matrix.to(device), captions.to(device)
                #detections, captions = detections, captions

                out = model(detections,ad_matrix, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss

#指标
def evaluate_metrics(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader)) as pbar:

        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)

            ad_matrix = caps_gt[0]
            ad_matrix = torch.cat(ad_matrix, dim=1).reshape(2, 10, 10).to(device)
            # print(ad_matrix)
            # print(ad_matrix.type)

            caps_gt = caps_gt[1]
            #print(caps_gt)

            #images = images

            with torch.no_grad():
                out, _ = model.beam_search(images,ad_matrix,20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores

#交叉熵损失训练
def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    scheduler.step()
    running_loss = .0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, ad_matrix,captions) in enumerate(dataloader):
            detections, ad_matrix, captions = detections.to(device),ad_matrix.to(device), captions.to(device)
            #detections, captions = detections, captions
            #print('!!!')
            #print(captions)

            out = model(detections,ad_matrix, captions) #(10,18,45)
            optim.zero_grad()
            captions_gt = captions[:, 1:].contiguous()


            out = out[:, :-1].contiguous()

            #labelsmoothing
            #loss_labelsmoothing = loss_ls_v2(out, captions_gt)
            #loss_labelsmoothing.backward()
            #optim.step()
            #this_loss = loss_labelsmoothing.item()
            #running_loss += this_loss

            #original(no labelsmoothing)
            loss = loss_fn(out.view(-1,len(text_field.vocab)),captions_gt.view(-1))
            loss.backward()
            optim.step()
            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)

    return loss


# class CELossWithLS(torch.nn.Module):
#     def __init__(self, classes=None, smoothing=0.1, gamma=3.0, isCos=True, ignore_index=-1):
#         super(CELossWithLS, self).__init__()
#         self.complement = 1.0 - smoothing
#         self.smoothing = smoothing
#         self.cls = classes
#         self.log_softmax = torch.nn.LogSoftmax(dim=1)
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#
#     def forward(self, logits, target):
#         with torch.no_grad():
#             oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).permute(0,1,2).contiguous()
#             smoothen_ohlabel = oh_labels * self.complement + self.smoothing / self.cls
#
#         logs = self.log_softmax(logits[target!=self.ignore_index])
#         pt = torch.exp(logs)
#         return -torch.sum((1-pt).pow(self.gamma)*logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()


#RL
def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    model.train()
    running_loss = .0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, (detections, caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            #detections = detections
            matrix = caps_gt[0]
            matrix = torch.cat(matrix, dim=1).reshape(2, 10, 10).to(device)   #(b_s,len(matrix))
            caps_gt = caps_gt[1]
            outs, log_probs = model.beam_search(detections,matrix, seq_len, text_field.vocab.stoi['<eos>'],
                                                beam_size, out_size=beam_size)
            optim.zero_grad()

            # Rewards
            caps_gen = text_field.decode(outs.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
            #reward = torch.from_numpy(reward).view(detections.shape[0], beam_size)

            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)

            loss = loss.mean()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline

if __name__ == '__main__':

    device = torch.device("cuda:0")
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--m', type=int, default=48)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--features_path', default='/data/zfzhu/lc/m2transformer/features/instruments18_caption/')
    #parser.add_argument('--features_path', default='E:/m2transformer/features/instruments18_caption/')
    parser.add_argument('--annotation_folder', type=str, default = 'annotations/annotations')
    parser.add_argument('--logs_folder', type=str, default='tensorboard_logs')
    args = parser.parse_args()
    print(args)

    print('Training')


    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=10, load_in_tmp=False)

    ad_matrix_field = MatrixField(detections_path=args.features_path, max_detections=10,load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field,ad_matrix_field,text_field, args.features_path, args.annotation_folder, args.annotation_folder)

    train_dataset, val_dataset = dataset.splits

    print("-"*100)
    print(len(train_dataset))   #1124
    print(len(val_dataset))     #392


    if not os.path.isfile('vocab_%s.pkl' % args.exp_name):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=2)
        pickle.dump(text_field.vocab, open('vocab_%s.pkl' % args.exp_name, 'wb'))
    else:
        text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    print(len(text_field.vocab))
    print(text_field.vocab.stoi)

    memory = np.load('memory48.npy')
    memory = memory[np.newaxis,:]

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, memory,attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': args.m})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])

    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)





    dict_dataset_train = train_dataset.image_dictionary({'image': image_field,'ad_matrix':ad_matrix_field, 'text': RawField()})
    #print(len(dict_dataset_train))  #1124

    ref_caps_train = list(train_dataset.text)
    ref_matrix_train = list(train_dataset.ad_matrix)
    ref_image_train = list(train_dataset.image)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))

    dict_dataset_val = val_dataset.image_dictionary({'image': image_field,'ad_matrix':ad_matrix_field, 'text': RawField()})
    #print(len(dict_dataset_val))   #392
    ref_caps_val = list(val_dataset.text)
    ref_matrix_val = list(val_dataset.ad_matrix)
    ref_image_val = list(val_dataset.image)


    def lambda_lr(s):
        warm_up = args.warmup
        s += 1
        return (model.d_model ** -.5) * min(s ** -.5, s * warm_up ** -1.5)


    # Initial conditions
    optim = Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98))
    scheduler = LambdaLR(optim, lambda_lr)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    patience = 0
    start_epoch = 0
    best_epoch = 0



    if args.resume_last or args.resume_best:
        if args.resume_last:
            fname = 'saved_models/%s_last_r6.pth' % args.exp_name
        else:
            fname = 'saved_models/%s_best_r6.pth' % args.exp_name

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))



    print("Training starts")
    #print(start_epoch)

    for e in range(start_epoch, start_epoch+100):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)


        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,drop_last=True)

        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                           num_workers=args.workers,drop_last=True)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5,drop_last=True)


        # train model with a word-level cross-entropy loss(xe)
        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)



        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        writer.add_scalar('data/val_loss', val_loss, e)

        # Validation scores
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)

        val_cider = scores['CIDEr']

        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_bleu = scores['BLEU'][0]
            best_cider = val_cider
            best_epoch = e
            best = True
        else:
            patience += 1

        print('patiece')
        print(patience)

        switch_to_rl = False
        exit_train = False


        if patience == 10:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim = Adam(model.parameters(), lr=5e-6)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True


        if switch_to_rl and not best:
            data = torch.load('saved_models/%s_best.pth' % args.exp_name)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'])
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        print("Validation scores", scores, 'Best epoch',best_epoch,'Best bleu:%.4f, cider:%.4f'%(best_bleu,best_cider))


        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, 'saved_models/%s_last.pth' % args.exp_name)

        if best:
            print('saving best epoch...!')
            copyfile('saved_models/%s_last.pth' % args.exp_name, 'saved_models/%s_best.pth' % args.exp_name)

        if exit_train:
            writer.close()
            break

