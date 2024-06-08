# %% [markdown]
# <a href="https://colab.research.google.com/github/natgluons/GNNs_HGTmodel/blob/main/HGTmodel_GNNs_fraud_nonfraud_paper.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## Library & Installation

# %%
install_deps = False
if install_deps:
    %pip install torch==2.2.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    %pip install cudatoolkitP
    %pip install umap-learn
    %pip install torch_geometric
    %pip install google-cloud-bigquery
    %pip install db-dtypes
    # %pip install kneed

# %%
runtime = "local"

# %%
# user authentication
if runtime == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
    from google.colab import auth
    auth.authenticate_user()

# %%
import umap
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from datetime import datetime
from torch_geometric.nn import HGTConv
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# BIGQUERY
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas_gbq

# DEBUGGING
# import warnings
# from sklearn.exceptions import ConvergenceWarning

# VISUALIZATION
# import numpy as np
# import matplotlib.pyplot as plt

# MACHINE LEARNING TRAINING
# from torch_geometric.utils import to_networkx
# from torch_geometric.data import Data
# from torch.optim import Adam
# from torch.nn.functional import log_softmax, nll_loss
# from kneed import KneeLocator
# from sklearn.metrics.pairwise import euclidean_distances
# from collections import defaultdict
# from tabulate import tabulate

# %%
program_start = datetime.now()
def elapsed():
    return datetime.now()-program_start

# %% [markdown]
# ## Query

# %%
query_cache = None
def query_(project_id):
  query = """
  -- use query based on your own dataset
  """

  if runtime == "colab":
    df = pd.read_gbq(query, project_id=project_id, dialect='standard')
  else:
    client = bigquery.Client(project=project_id)
    query_job = client.query(query)
    df = query_job.to_dataframe()
    
  return df

# %% [markdown]
# ## Preprocess

# %%
def preprocess(df):
	# Separating nodes and edges
	node = df[['user_id', 'type', 'province', 'type_desc', 'business_segment', 'business_sub_segment', 'account_age_months', 'outbound_trx', 'inbound_trx', 'hit_count', 'reported_risk', 'data_type']]
	nodes = node[node['data_type'] == 'node'].drop('data_type', axis=1)

	edges = df[df['data_type'] == 'edge'].drop(['data_type'] + list(nodes.columns.difference(['source_id', 'target_id'])), axis=1)

	nodes['user_id'] = nodes['user_id'].astype('int64')
	nodes['account_age_months'] = nodes['account_age_months'].astype('int64')
	nodes['outbound_trx'] = nodes['outbound_trx'].astype('int64')
	nodes['inbound_trx'] = nodes['inbound_trx'].astype('int64')
	nodes['hit_count'] = nodes['hit_count'].astype('int64')
	nodes['reported_risk'] = nodes['reported_risk'].astype('int64')

	edges['source_id'] = edges['source_id'].astype('int64')
	edges['target_id'] = edges['target_id'].astype('int64')
	edges['trans_amount'] = edges['trans_amount'].astype('int64')

	#  Create a unified index mapping
	all_ids = pd.concat([nodes['user_id'], edges['source_id'], edges['target_id']]).unique()
	id_to_index = {id: idx for idx, id in enumerate(all_ids)}

	# Make a new column to replace original IDs with indices
	nodes['userid'] = nodes['user_id'].map(id_to_index)
	edges['sourceid'] = edges['source_id'].map(id_to_index)
	edges['targetid'] = edges['target_id'].map(id_to_index)

	return nodes, edges

# %%
def encode(nodes, edges):
    ### Preprocessing for HGT ###

    # Encode categorical attributes for nodes
    encoder = LabelEncoder()
    for col in ['type', 'province', 'type_desc', 'business_segment', 'business_sub_segment']:
        nodes[col + '_code'] = encoder.fit_transform(nodes[col])
    node_features = torch.tensor(nodes[['type_code', 'province_code', 'type_desc_code', 'business_segment_code', 'business_sub_segment_code', 'account_age_months', 'inbound_trx', 'outbound_trx']].values, dtype=torch.float)
    # node_labels = torch.tensor(nodes['reported_risk'].values, dtype=torch.long)
    # labeled_mask = node_labels != 0

    edge_index = torch.tensor(edges[['sourceid', 'targetid']].values.T, dtype=torch.long)
    edge_index[edge_index < 0] = 0
    # edge_attr = torch.tensor(edges[['trans_amount']].values, dtype=torch.float)
    edge_type = torch.zeros(edges.shape[0], dtype=torch.long)

    return node_features, edge_index, edge_type

# %% [markdown]
# ## Model

# %%
### HGT Model Training & Clustering ###

class HGT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_node_types, num_relations):
        super(HGT, self).__init__()
        self.node_type_embedding = torch.nn.Embedding(num_embeddings=num_node_types, embedding_dim=in_channels)
        self.edge_type_embedding = torch.nn.Embedding(num_embeddings=num_relations, embedding_dim=in_channels)
        
        self.conv1 = HGTConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = HGTConv(hidden_channels, out_channels, num_relations=num_relations)

    def forward(self, x, edge_index, edge_type, node_type):
        x = x + self.node_type_embedding(node_type)  # Optional: Enhance node features with type embeddings
        edge_type = self.edge_type_embedding(edge_type)  # Optional: Enhance edge features with type embeddings

        x = F.relu(self.conv1(x, edge_index, edge_type=edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type=edge_type))
        return x

# %%
def load_model(model_path):
    model = HGT(in_channels=8, hidden_channels=16, out_channels=3, num_node_types=4, num_relations=2)
    model.load_state_dict(torch.load(model_path))
    return model

# %%
def predict(model, node_features, edge_index, edge_type):
	###  Load Trained Model and Use It for Inference ###
	print(elapsed(), "model eval...")
	model.eval()
	print(elapsed(), "finished model eval")

	# Generate embeddings
	print(elapsed(), "generating embeddings...")
	with torch.no_grad():
		embeddings = model(node_features, edge_index, edge_type)
	print(elapsed(), "finished generating embeddings")

	# Use UMAP to project embeddings to two dimensions for visualization
	print(elapsed(), "running UMAP transform...")
	UMAP = umap.UMAP(n_components=2)
	embeddings_2d = UMAP.fit_transform(embeddings)
	print(elapsed(), "finished running UMAP transform")

	# Apply K-Means to the UMAP 2D embeddings
	print(elapsed(), "applying k means...")
	kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
	clusters = kmeans.fit_predict(embeddings_2d)
	print(elapsed(), "finished applying k means")

	return clusters

# %%
def make_graph(edge_index, edges):
    # Create a NetworkX graph from the edge_index tensor
    # G = nx.DiGraph()  # Use DiGraph to calculate in-degree and out-degree separately
    print(elapsed(), "adding edges...")
    G = nx.from_pandas_edgelist(edges, 'sourceid', 'targetid', create_using=nx.DiGraph())
    print(elapsed(), "edges:", edges)
    print(elapsed(), "finished adding edges")
    print(elapsed(), "graph nodes:", G.nodes())
    print(elapsed(), "graph edges:", G.edges())
    edge_index_np = edge_index.cpu().numpy()

    # Calculate network metrics
    print(elapsed(), "calculating network metrics...")
    print(elapsed(), "calculating degree centrality...")
    degree_centrality = nx.degree_centrality(G)
    print(elapsed(), "finished calculating degree centrality")
    print(elapsed(), "calculating indegree centrality...")
    in_degree_centrality = nx.in_degree_centrality(G)
    print(elapsed(), "finished calculating indegree centrality")
    print(elapsed(), "calculating outdegree centrality...")
    out_degree_centrality = nx.out_degree_centrality(G)
    print(elapsed(), "finished calculating outdegree centrality")
    print(elapsed(), "calculating betweenness centrality...")
    betweenness_centrality = nx.betweenness_centrality(G)
    print(elapsed(), "finished calculating betweenness centrality")
    print(elapsed(), "finished calculating network metrics")

    # Convert centrality measures to DataFrame
    centrality_df = pd.DataFrame({
        'userid': list(degree_centrality.keys()),
        'degree_centrality': list(degree_centrality.values()),
        'in_degree_centrality': list(in_degree_centrality.values()),
        'out_degree_centrality': list(out_degree_centrality.values()),
        'betweenness_centrality': list(betweenness_centrality.values())
    })

    return centrality_df

# %%
def identify_syndicate(nodes, centrality_df):
    # Normalize centrality measures to range between 0 and 1
    centrality_df['normalized_in_degree'] = centrality_df['in_degree_centrality'] / centrality_df['in_degree_centrality'].max()
    centrality_df['normalized_out_degree'] = centrality_df['out_degree_centrality'] / centrality_df['out_degree_centrality'].max()
    centrality_df['betweenness_centrality'] = centrality_df['betweenness_centrality'] / centrality_df['betweenness_centrality'].max()

    # Define weights for each centrality measure
    weights = {
        'in_degree': 0.4,
        'out_degree': 0.4,
        'betweenness': 0.2
    }

    # Calculate the weighted syndicate score
    centrality_df['syndicate_score'] = (weights['in_degree'] * centrality_df['normalized_in_degree'] +
                                        weights['out_degree'] * centrality_df['normalized_out_degree'] +
                                        weights['betweenness'] * centrality_df['betweenness_centrality'])

    # Scale the score to be between 0 and 1
    max_score = centrality_df['syndicate_score'].max()
    if max_score > 0:  # Avoid division by zero
        centrality_df['syndicate_score'] /= max_score

    # Assuming 'nodes' is your main DataFrame containing user details and is globally accessible or passed as a parameter
    nodes = pd.merge(nodes, centrality_df[['userid', 'syndicate_score']], on='userid', how='left')
    print("centrality df head:", centrality_df.head(10))
    print("centrality df userid max:", centrality_df['userid'].max())
    print("nodes userid max:", nodes['userid'].max())
    print("nodes columns:", nodes.columns)
    print("nodes head:", nodes.head(10))

    return nodes

# %%
# heuristic risk mapping for multiple entries
def map_heuristic_risk(province, type_desc, business_segment, business_sub_segment):
    # Define the heuristic risk mapping
    high_risk_values = ('3') # specified pre-defined high risk value (internal rules)
    medium_risk_values = ('2') # specified pre-defined high risk value (internal rules)

    # Check if the input values belong to high or medium risk category
    if province in high_risk_values or type_desc in high_risk_values or business_segment in high_risk_values or business_sub_segment in high_risk_values:
        return 'high'
    elif province in medium_risk_values or type_desc in medium_risk_values or business_segment in medium_risk_values or business_sub_segment in medium_risk_values:
        return 'medium'
    else:
        return 'low'

# %%
# Heuristic thresholds and mappings
hit_count_thresholds = {'high': 332, 'medium': 3}

# Function to calculate individual node risk scores
def calculate_node_risk_score(row):
    # Primary factor: Syndicate Score directly influences the base score
    # Assuming syndicate_score is normalized between 0 and 1
    base_score = row['syndicate_score']

    # Secondary adjustments
    # Hit count risk - smaller impact
    if row['hit_count'] > hit_count_thresholds['high']:
        base_score += 0.1  # Small adjustment for extreme cases
    elif row['hit_count'] > hit_count_thresholds['medium']:
        base_score += 0.05  # Even smaller adjustment for medium cases

    # heuristic risk - using a mapped function to get risk adjustment
    heuristic_adjustment = map_heuristic_risk(row['province'], row['type_desc'], row['business_segment'], row['business_sub_segment'])
    if heuristic_adjustment == 'high':
        base_score += 0.1  # heuristic issues can adjust the score slightly
    elif heuristic_adjustment == 'medium':
        base_score += 0.05

    # Cap the score at 1.0 since the scale is 0 to 1
    final_score = min(base_score, 1.0)
    return final_score

# %%
def calculate_risk_scores(nodes):
	# Calculate risk scores for each node
	nodes['risk_index'] = nodes.apply(calculate_node_risk_score, axis=1)

	# Normalize the risk index using Min-Max scaling
	min_risk_index = nodes['risk_index'].min()
	max_risk_index = nodes['risk_index'].max()
	nodes['risk_index'] = (nodes['risk_index'] - min_risk_index) / (max_risk_index - min_risk_index)

	# # Aggregate risk scores by HGT_cluster to get the average risk score for each cluster
	cluster_risk_scores = nodes.groupby('HGT_cluster')['risk_index'].mean().reset_index()
	cluster_risk_scores.rename(columns={'risk_index': 'cluster_risk_score'}, inplace=True)

	# Calculate the thresholds for 'low' and 'high' risk levels based on quantiles
	low_threshold = cluster_risk_scores['cluster_risk_score'].quantile(0.33)
	high_threshold = cluster_risk_scores['cluster_risk_score'].quantile(0.67)

	# Define function to assign risk levels based on quantile thresholds
	def assign_risk_level(final_score):
		if final_score >= high_threshold:
			return 'high'
		elif final_score > low_threshold:
			return 'medium'
		else:
			return 'low'

	# Assign risk levels based on quantiles
	cluster_risk_scores['cluster_risk_level'] = cluster_risk_scores['cluster_risk_score'].apply(assign_risk_level)
	cluster_risk_scores['syndicate_score'] = nodes['syndicate_score']
	cluster_risk_scores['node_count'] = nodes.groupby('HGT_cluster')['HGT_cluster'].transform('count')

	nodes = nodes.merge(cluster_risk_scores, on='HGT_cluster', how='left')

	return nodes, cluster_risk_scores

# %%
# Set adjustment factors
degree_adjustment_factor = 0.05
betweenness_adjustment_factor = 0.10
def calculate_risk_index(cluster_risk_score, degree_centrality, betweenness_centrality):
    # Adjust the risk index for each merchant node based on centrality measures
    adjusted_risk_index = cluster_risk_score + \
        (degree_centrality * degree_adjustment_factor) + \
        (betweenness_centrality * betweenness_adjustment_factor)

    # Normalize the adjusted risk index using Min-Max scaling
    min_risk_index = adjusted_risk_index.min()
    max_risk_index = adjusted_risk_index.max()
    risk_index = (adjusted_risk_index - min_risk_index) / (max_risk_index - min_risk_index)

    # Calculate quantile thresholds for 'low' and 'high' risk levels
    low_quantile = risk_index.quantile(0.33)
    high_quantile = risk_index.quantile(0.67)

    # Define function to assign merchant risk levels based on quantile thresholds
    def assign_risk_level(score):
        if score >= high_quantile:
            return 'high'
        elif score > low_quantile:
            return 'medium'
        else:
            return 'low'

    # Assign risk levels based on quantiles
    risk_level = risk_index.apply(assign_risk_level)

    return adjusted_risk_index, risk_index, risk_level

# %% [markdown]
# ## Workflow

# %%
def load_data_and_process(project_id):
    print(elapsed(), "running query...")
    # df = query(project_id) # uncomment this line to run the query
    df = pd.read_csv("./gnn_fraud_sampling.csv")
    print(elapsed(), "finished running query")

    # Assuming preprocess is a function that prepares your data
    print(elapsed(), "preprocessing...")
    nodes, edges = preprocess(df)
    print(elapsed(), "finished preprocessing")

    # Assuming encode is a function that extracts features and encodes them for the model
    print(elapsed(), "encoding...")
    node_features, edge_index, edge_type = encode(nodes, edges)
    print(elapsed(), "finished encoding")

    return nodes, edges, node_features, edge_index, edge_type

def workflow(model_path, nodes, edges, node_features, edge_index, edge_type):
    # Load the predictive model
    print(elapsed(), "loading model...")
    model = load_model(model_path)
    print(elapsed(), "finished loading model")

    # Predict cluster memberships
    print(elapsed(), "predicting...")
    nodes['HGT_cluster'] = predict(model, node_features, edge_index, edge_type)
    print(elapsed(), "finished predicting")
    # Generate graph and calculate centrality metrics

    print(elapsed(), "making graph...")
    centrality_df = make_graph(edge_index, edges)
    nodes = nodes.merge(centrality_df, on='userid', how='left')
    nodes['degree_centrality'].fillna(0, inplace=True)
    nodes['betweenness_centrality'].fillna(0, inplace=True)
    print(elapsed(), "finished making graph")

    # Identify syndicate nodes using the centrality data
    print(elapsed(), "identifying syndicates...")
    nodes = identify_syndicate(nodes, centrality_df)
    print(elapsed(), "finished identifying syndicates")

    # Calculate risk scores incorporating new syndicate scores
    print(elapsed(), "calculating risk scores...")
    nodes, cluster_risk_scores = calculate_risk_scores(nodes)
    print(elapsed(), "finished calculating risk scores")

    # Aggregate users within each cluster
    clustered_users = nodes.groupby('HGT_cluster')['userid'].apply(list).reset_index()
    clustered_users['userid'] = clustered_users['userid'].apply(lambda x: ', '.join(map(str, x)))

    # Assuming calculate_risk_index is a function to adjust risk indices based on new metrics
    adjusted_risk_index, risk_index, risk_level = calculate_risk_index(
        nodes['cluster_risk_score'], nodes['degree_centrality'], nodes['betweenness_centrality'])

    nodes['adjusted_risk_index'] = adjusted_risk_index
    nodes['risk_index'] = risk_index
    nodes['risk_level'] = risk_level

    attributesetc = None
    return nodes, cluster_risk_scores, clustered_users, attributesetc

# %%
def modify_nodes_df(nodes):
    # Create new columns by aggregating other columns into dictionaries
    nodes['attributes'] = nodes[['account_age_months', 'degree_centrality',
                                      'in_degree_centrality', 'out_degree_centrality',
                                      'betweenness_centrality']].apply(lambda x: x.to_dict(), axis=1)

    # Drop specified columns
    nodes.drop(columns=['userid', 'type_code', 'province_code', 'type_desc_code',
                        'business_segment_code', 'business_sub_segment_code'], inplace=True)

    # Prepare final DataFrame
    final_result = nodes[['user_id', 'type', 'attributes', 'syndicate_score', 'HGT_cluster', 'cluster_risk_level', 'risk_level', 'cluster_risk_score']]
    final_result = final_result.sort_values(by='syndicate_score', ascending=False)

    return final_result

# %% [markdown]
# ## Main to result

# %%
# Main execution
program_start = datetime.now()
print("program started at:", program_start)
project_id = "project-linkaja-dataset"
model_path = "./models/HGT_model.pth"
if runtime == "colab":
    model_path = "/content/drive/My Drive/HGT_model.pth"

# Load data and prepare it for the model
nodes, edges, node_features, edge_index, edge_type = load_data_and_process(project_id)

# Run the model and calculate risk scores
nodes, cluster_risk_scores, clustered_users, attributesetc = workflow(
    model_path, nodes, edges, node_features, edge_index, edge_type)

final_result = modify_nodes_df(nodes)

# Print results
print("final_result: ", final_result, "\n")
print("numeric_stats: ", final_result.describe(), "\n")
print("cluster_risk_scores: ", cluster_risk_scores, "\n")

# End program
print("program ended at:", datetime.now())


