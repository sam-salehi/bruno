
import random
import torch
import numpy as np
seed = 345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from sklearn.decomposition import PCA


import torch 
import os 
import pandas as pd 
from bruno.nn.modules import Encoder
from bruno.learn import Hyperparameters
from bruno.learn import TrainModel
from torch_geometric.transforms import RandomNodeSplit
import anndata as ad
from bruno.data import AnnDataToGraphData
from bruno.data import PrepareAnnData
from bruno.data import get_map
from bruno.data import ReactomeNetwork,Reactome


from bruno.data import ReactomeNetwork,Reactome




from bruno.data import Reactome, ReactomeNetwork

reactome_base_dir = 'data/reactome'#updated
relations_file_name = 'ReactomePathwaysRelation.txt'
pathway_names = 'ReactomePathways.txt'
pathway_genes = 'ReactomePathways.gmt'

reactome = Reactome(reactome_base_dir,
                    relations_file_name,
                    pathway_names,
                    pathway_genes)
names_df = reactome.pathway_names
hierarchy_df = reactome.hierarchy
genes_df = reactome.pathway_genes

reactome_net = ReactomeNetwork(reactome_base_dir,
                    relations_file_name,
                    pathway_names,
                    pathway_genes)


obs_vars = ['Purity', 'Ploidy', 'Tumor.Coverage', 'Normal.Coverage', 'Mutation.burden', 'Fraction.genome.altered', 'Mutation_count']
obs_vars = obs_vars.append("response")

prepareData = PrepareAnnData(data = all, obs_vars=obs_vars, map = get_map(reactome_net, n_levels=3))
adata, map = prepareData.anndata()
demo = AnnDataToGraphData("data",
        transform=RandomNodeSplit(split="random", num_train_per_class = 200, num_val=200, num_test=314), 
        group = 'response',
        adata = adata,
        knn = 3)
data = demo.data

args = Hyperparameters()
args.epochs = 2000
args.num_node_features = data.num_node_features
args.num_classes = len(data.y.unique())
args.cuda = args.cuda and torch.cuda.is_available() 

args.method = "ANN"
ann = Encoder(map, args=args, bias = False)
train_ann = TrainModel(model=ann, graph=data, args=args)

