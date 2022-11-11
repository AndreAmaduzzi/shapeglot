import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.utils.data import Dataset

from shapeglot.simple_utils import unpickle_data
from shapeglot.in_out.rnn_data_preprocessing import make_dataset_for_rnn_based_model
from shapeglot.in_out.shapeglot_dataset import ShapeglotDataset
from shapeglot.in_out.geometry import vgg_image_features, pc_ae_features
from shapeglot.models.neural_utils import MLPDecoder, smoothed_cross_entropy
from shapeglot.models.encoders import LanguageEncoder, PretrainedFeatures
from shapeglot.models.listener import Listener
from shapeglot.vis_utils import visualize_example
from shapeglot import vis_utils

def main():
    ## PREPARE DATA FOR TRAINING => USE TEXT2SHAPE
    top_data_dir = './data/main_data_for_chairs'
    top_pretrained_feat_dir = './data/main_data_for_chairs/pretrained_features'
    top_image_dir = osp.join(top_data_dir, 'images/shapenet/')
    vis_utils.top_image_dir = top_image_dir
    # Load game-data
    data_name = 'game_data.pkl'
    game_data, word_to_int, int_to_word, int_to_sn_model, sn_model_to_int, sorted_sn_models =\
    unpickle_data(osp.join(top_data_dir, data_name))

    # Load pre-trained 2D/3D features.
    vgg_feats_file = osp.join(top_pretrained_feat_dir, 'shapenet_chair_vgg_fc7_embedding.pkl')
    vgg_feats = vgg_image_features(int_to_sn_model, 'chair', vgg_feats_file, python2_to_3=True)

    pc_feats_file = osp.join(top_pretrained_feat_dir, 'shapenet_chair_pcae_128bneck_chamfer_embedding.npz')
    pc_feats = pc_ae_features(int_to_sn_model, pc_feats_file)

    max_seq_len = 33              # Maximum size (in tokens) per utterance.
    split_sizes = [0.8, 0.1, 0.1] # Train-val-test sizes.
    random_seed = 2004
    unique_test_geo = True        # If true, the test/train/val splits have 'targets' that are disjoint sets.
    only_correct = True           # Drop all not correctly guessed instances.

    net_data, split_ids, _, net_data_mask = make_dataset_for_rnn_based_model(game_data, 
                                                                            split_sizes, 
                                                                            max_seq_len,
                                                                            drop_too_long=True,
                                                                            only_correct=only_correct,
                                                                            unique_test_geo=unique_test_geo,
                                                                            replace_not_in_train=True,
                                                                            geo_condition=None, 
                                                                            bias_train=False, 
                                                                            seed=random_seed)
                                                                            
    dataloaders = {}
    num_workers = 8
    batch_size = 2048 # Yes this is a big batch-size.

    for split in ['train', 'val', 'test']:
        dataset = ShapeglotDataset(net_data[split])
        print(f'split: {split}, dataset size: {len(dataset)}')
        dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=batch_size,
                                                        shuffle=split=='train',
                                                        num_workers=num_workers)
        
    ## DEFINE MODELS
    embedding_dim = 100  # Dimension of INPUT space of LSTM
    n_hidden = 100       # LSTM hidden size
    vocab_size = len(int_to_word)
    train_epochs = 100
    learning_rate = 0.005
    reg_gamma = 0.005    # Weight regularization on FC layers

    visual_encoder = PretrainedFeatures(torch.Tensor(vgg_feats), embed_size=embedding_dim)
    pc_encoder = PretrainedFeatures(torch.Tensor(pc_feats), embed_size=embedding_dim)   # use PointNetEncoder instead of this   
    lang_encoder = LanguageEncoder(n_hidden, embedding_dim, vocab_size)                 # use CLIP instead of this
    mlp_decoder = MLPDecoder(n_hidden*2, [100, 50, 1], True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    listener = Listener(lang_encoder, visual_encoder, mlp_decoder, pc_encoder).to(device)
    optimizer = optim.Adam(listener.parameters(), lr=learning_rate)

    ## TRAIN THE MODEL
    best_val_accuracy = 0.0
    best_test_accuracy = 0.0

    for epoch in range(1, train_epochs+1):
        for phase in ['train', 'val', 'test']:
            if phase == 'train':            
                listener.train()
            else:
                listener.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for context_ids, targets, tokens in dataloaders[phase]:
                context_ids = context_ids.to(device)    # context_ids size: [B, 3]
                targets = targets.to(device)            # targets size: [B]
                tokens = tokens.to(device)              # tokens size: [B, 34]       
        
                with torch.set_grad_enabled(phase == 'train'):
                    logits = listener(context_ids, tokens)  # logits size: [B, 3] => prediction for each shape
                    loss = smoothed_cross_entropy(logits, targets)                                
                    reg_loss = 0.0
                    
                    for p in visual_encoder.named_parameters():
                        if 'fc.weight' == p[0]:
                            reg_loss += p[1].norm(2)
                    
                    for p in pc_encoder.named_parameters():
                        if 'fc.weight' == p[0]:
                            reg_loss += p[1].norm(2)
                            
                    reg_loss *= reg_gamma                
                    loss += reg_loss
                    
                    _, preds = torch.max(logits, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        listener.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # statistics
                running_loss += loss.item() * targets.size(0)
                running_corrects += torch.sum(preds == targets)
            
            n_examples = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / n_examples
            epoch_acc = running_corrects.double() / n_examples
            
            if phase == 'val':
                if epoch_acc > best_val_accuracy:
                    best_val_accuracy = epoch_acc
                    val_improved = True
                else:
                    val_improved = False
            
            if phase == 'test' and val_improved:
                best_test_accuracy = epoch_acc
                            
            print('{} Epoch {} Loss: {:.4f} Acc: {:.4f}'.format(epoch, phase, epoch_loss, epoch_acc))
        print('Best test so far: {:.4f}'.format(best_test_accuracy))


if __name__ == '__main__':
    main()