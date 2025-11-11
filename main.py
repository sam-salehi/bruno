
import os
import torch
from data_utils import load_and_merge_tables, build_reactome_network, prepare_graph_data
from training import train_models
from explain_runner import explain_and_save
from bruno.nn.modules import Encoder
from bruno.learn import Hyperparameters, TrainModel
import pandas as pd 




def main() -> None:
    data_dir = "data"
    reactome_dir = os.path.join(data_dir, "reactome")
    outputs_root = "./outputs"

    clin_vars = ['Purity', 'Ploidy', 'Tumor.Coverage', 'Normal.Coverage', 'Mutation.burden', 'Fraction.genome.altered', 'Mutation_count']
    obs_vars = clin_vars.copy()
    obs_vars.append('response')

    merged = load_and_merge_tables(data_dir)
    reactome_net = build_reactome_network(reactome_dir)
    data, adata, map_df = prepare_graph_data(merged, obs_vars, reactome_net, data_dir)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    args = Hyperparameters()
    args.epochs = 2000
    args.num_node_features = data.num_node_features
    args.num_classes = int(len(data.y.unique()))
    args.cuda = (device.type == "cuda")
    args.device = device


    # ann = Encoder(map_df,args=args,bias=False)
    # train_ann = TrainModel(model=ann,graph=data,args=args)

    # print(train_ann.metrics())
    # train_ann.plot_loss()
   
    models = train_models(data, map_df, device=device, epochs=2000, patience=50, save_dir=".")

    # explain_and_save(models, data, device=device, outputs_root=outputs_root, return_type="log_probs", epochs=200, max_nodes=None)


if __name__ == "__main__":
    main()


