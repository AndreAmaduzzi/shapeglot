import torch
import numpy as np
import os.path as osp
import torch.nn.functional as F
import sys
sys.path.append("../Infusion/dataset")
from text2shape_dataset import Text2Shape

from torch import nn
from torch import optim
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from shapeglot.simple_utils import unpickle_data
from shapeglot.in_out.rnn_data_preprocessing import make_dataset_for_rnn_based_model
from shapeglot.in_out.shapeglot_dataset import ShapeglotDataset, T2SShapeglotDataset, shuffle_ids
from shapeglot.in_out.geometry import vgg_image_features, pc_ae_features
from shapeglot.models.neural_utils import MLPDecoder, smoothed_cross_entropy
from shapeglot.models.encoders import LanguageEncoder, PretrainedFeatures, PointCloudEncoder
from shapeglot.models.listener import Listener, T2S_Listener
from shapeglot.vis_utils import visualize_example
from shapeglot import vis_utils

def visualize_data_sample(pointclouds, target, text, path):
    n_clouds = len(pointclouds)
    fig = plt.figure(figsize=(20,20))
    plt.title(text)
    plt.axis('off')
    ncols = n_clouds
    nrows = 1
    for idx, pc in enumerate(pointclouds):
        colour = 'r' if target == idx else 'b'
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=5)
        ax.view_init(elev=30, azim=255)
        ax.axis('off')
    plt.savefig(path)
    plt.close(fig)


def main():
    split_sizes = [0.8, 0.1, 0.1] # Train-val-test sizes.
    random_seed = 2004
    unique_test_geo = True        # If true, the test/train/val splits have 'targets' that are disjoint sets.
    only_correct = True           # Drop all not correctly guessed instances.

    dataloaders = {}
    num_workers = 8
    batch_size = 2048       # Yes this is a big batch-size.
    max_length = 77         # max length of text embedding sequence, to truncate when loading the dataset

    dataroot = "/media/data2/aamaduzzi/datasets/Text2Shape/"
    for split in ["test"]:
        dataset = Text2Shape(root=Path(dataroot),
                                    split=split,
                                    categories="all",
                                    from_shapenet_v1=True,
                                    from_shapenet_v2=False,
                                    conditional_setup=True,
                                    language_model="t5-11b",
                                    lowercase_text=True,
                                    max_length=max_length,
                                    padding=False,
                                    scale_mode="global_unit")

        dataloaders[split] = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=batch_size,
                                                        shuffle=split=='train',
                                                        num_workers=num_workers)
    
    phase = "test"
    for sample in dataloaders[phase]:   # sample is a dict with <batch size> elements
        print('data batch is here')
        pairs=[]
        start = datetime.now()
        for idx, target_model_id in enumerate(tqdm(sample["model_id"])):
            # for each element I pick one other random model_id WITHIN CATEGORY => I get pairs of chairs together and pairs of tables together
            shape_class = sample["cate"][idx]
            # pick randomly another chair
            sample_ids = np.array(sample["model_id"])                       # all model_ids of the batch
            idxs = np.where((sample_ids != target_model_id))[0] # pick elements with same object class and different model_id
            
            # first choice
            chosen_idx = np.random.choice(idxs) # choose a random idx among the accepted ones
            distr_model_id = sample["model_id"][chosen_idx]
            text_embed = sample["text"][idx]    # get text embedding of the current sentence

            model_ids = [target_model_id, distr_model_id]
            target = 0
            target_cloud = sample["pointcloud"][idx]
            distr_cloud = sample["pointcloud"][chosen_idx]
            clouds = [target_cloud, distr_cloud]
            # shuffle the order of the model_ids and the corresponding target
            clouds, target = shuffle_ids(clouds, target)

            pairs.append({
                "clouds": clouds,
                "target": target,
                "text_embed": text_embed    
            })

            #visualize_data_sample(clouds, target, text_embed, f"sample_{datetime.now()}.png")

            # second choice
            chosen_idx_2 = chosen_idx
            while chosen_idx_2 == chosen_idx:
                chosen_idx_2 = np.random.choice(idxs) # choose a random idx among the accepted ones
            distr_model_id = sample["model_id"][chosen_idx_2]
            model_ids = [target_model_id, distr_model_id]
            target = 0
            distr_cloud = sample["pointcloud"][chosen_idx]
            clouds = [target_cloud, distr_cloud]
            # shuffle the order of the model_ids and the corresponding target
            clouds, target = shuffle_ids(clouds, target)
    
            pairs.append({
                "clouds": clouds,
                "target": target,
                "text_embed": text_embed
            })

            #visualize_data_sample(clouds, target, text_embed, f".sample_{datetime.now()}.png")

        end = datetime.now()
        print('data is ready')
        print('time for building discrimination pairs in a batch: ', (end-start).total_seconds() ,' seconds')
        print('n of pairs: ', len(pairs))


    ## DEFINE MODELS
    #embedding_dim = TODO: to be set
    train_epochs = 100
    learning_rate = 0.005
    reg_gamma = 0.005    # Weight regularization on FC layers

    ckpt_path = ("./shapeglot/models/last.ckpt")
    pc_encoder = PointCloudEncoder(pointnet_path=ckpt_path, embed_size=1024)   # use PointNetEncoder instead of this   
    mlp_decoder = MLPDecoder(n_hidden*2, [100, 50, 1], True)    # originally, it is from 200 to 100, then 50, then 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    listener = T2S_Listener(mlp_decoder, pc_encoder).to(device)
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
            for sample in dataloaders[phase]:
                
                # for each model_id:
                    # I take two random elements which have model_id different from this one
                    # I build triplet with cloud_1, cloud_2, cloud_3, text_embed (where cloud_1 is the correct cloud)

                # prepare text embedding of each element
                text_embed = sample["text_embed"]
                cloud = sample["pointcloud"]
                # apply clever padding to text embeddings: truncate text embeddings to longest sequence in the batch
                sum_text_embed = torch.sum(text_embed, dim=2)
                max_seq_len = 0
                max_seq_len_idx = 0
                for idx, embed in enumerate(sum_text_embed):
                    zero_idx = torch.argmin(abs(embed), dim=0)
                    if zero_idx.item()>max_seq_len:
                        max_seq_len=zero_idx

                text_embed = text_embed[:,:max_seq_len, :]



                clouds = clouds.to(device)              # [B, 3, 3] => triplet of clouds
                text_embed = text_embed.to(device)      # [B, text_embed_shape] => embedding of the sentence
                target = target.to(device)              # [B] => index of the correct shape (0 or 1 or 2)
        
                with torch.set_grad_enabled(phase == 'train'):
                    logits = listener(context_ids, tokens)
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