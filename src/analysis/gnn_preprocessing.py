

"""
Designed to transform a standard employee dataset into a graph data structure, which is ideal for Graph Neural Networks (GNNs).
How It Works:

1. Building the Graph's Edges (build_edges)

This function creates the connections, or edges, between the employees in the dataset. It creates two types of edges:

Manager-Employee Edges: It iterates through the DataFrame to create directed edges from a manager's EmployeeID to each of their direct reports' EmployeeID. 
                        This represents the reporting hierarchy. The edge is labeled 'manages'.

Department Co-membership Edges: It groups employees by Department. 
                                Within each department, it creates undirected edges between every pair of employees. 
                                This represents a social or collaborative network, where people in the same department are connected. 
                                These edges are labeled 'same_dept'.
The function returns a list of tuples, where each tuple represents an edge: (source_node, target_node, edge_type).

2. Preparing Data for GNNs (export_for_gnn)
This function prepares the data in the format required by most GNN libraries.

Node Features: It creates a DataFrame of node features by dropping the Attrition column and setting EmployeeID as the index.
             The remaining columns (e.g., age, salary, department) become the attributes or features of each employee node.

Node Labels: It creates a numerical array of labels for the nodes. 
                It converts the categorical Attrition column ('Yes' and 'No') into a binary numerical format (1 for 'Yes', 0 for 'No'). 
                This is the target variable a GNN would try to predict.

"""


"""
Enhanced EmployeeGraphBuilder:
- Builds edges from multiple relationships (manager, department).
- Prepares node features and labels.
- Exports directly into PyTorch Geometric Data object.
"""
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

class EmployeeGraphBuilder:
    def __init__(self, df):
        self.df = df.copy()
        self.node_ids = {eid: idx for idx, eid in enumerate(self.df['EmployeeID'])}

    def build_edges(self, k=5):
        """
        Build edges for GNN graph:
        - Manager → Employee edges (directed, hierarchy).
        - Department-level k-nearest neighbor edges (undirected, sparser than full clique).

        Parameters:
        -----------
        k : int
            Number of nearest peers to connect per employee within each department.
            Reduces edge explosion compared to full all-to-all connections.
        """
        edges = []

        # --- Manager → Employee edges ---
        if 'EmployeeID' in self.df.columns and 'CurrentManager' in self.df.columns:
            for _, row in self.df.dropna(subset=['CurrentManager']).iterrows():
                manager = row['CurrentManager']
                emp = row['EmployeeID']
                if manager in self.node_ids and emp in self.node_ids:
                    edges.append((self.node_ids[manager], self.node_ids[emp], 'manages'))

        # --- Department k-NN peer edges ---
        if 'Department' in self.df.columns and 'EmployeeID' in self.df.columns:
            # pick numeric cols as "similarity features"
            num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            for dept, group in self.df.groupby('Department'):
                if len(group) <= 1:
                    continue
                emp_ids = group['EmployeeID'].tolist()
                X = group[num_cols].fillna(0).values

                # compute distance matrix inside department
                dist = euclidean_distances(X)

                for i, eid in enumerate(emp_ids):
                    # get k nearest peers (excluding self)
                    nearest = np.argsort(dist[i])[1:k+1]
                    for j in nearest:
                        peer_eid = emp_ids[j]
                        if eid in self.node_ids and peer_eid in self.node_ids:
                            # add undirected peer edges
                            edges.append((self.node_ids[eid], self.node_ids[peer_eid], 'same_dept'))
                            edges.append((self.node_ids[peer_eid], self.node_ids[eid], 'same_dept'))

        return edges


    def export_for_gnn(self, ignore_cols=None):
        """Prepare node features + labels for GNN libraries."""
        drop_cols = ['EmployeeID', 'Attrition']
        if ignore_cols:
            drop_cols += ignore_cols

        features = self.df.drop(columns=drop_cols, errors="ignore")

        # Convert all object columns to numeric codes
        for col in features.select_dtypes(include=['object']).columns:
            features[col] = features[col].astype('category').cat.codes

        # Fill any missing values
        features = features.fillna(0)

        # Labels: Attrition → 0/1
        labels = (self.df['Attrition'].astype(str).str.upper() == 'YES').astype(int).values
        return features, labels


    def to_pyg(self, k=5):
        """
        Convert employee dataset into a PyTorch Geometric Data object.
        - Builds edges (manager + department k-NN).
        - Exports node features + labels.
        - Prints graph summary for sanity checking.
        """
        # Prepare features and labels
        features, labels = self.export_for_gnn()
        
        # Build edges (with k-nearest department peers)
        edges = self.build_edges(k=k)

        # Convert edges into tensor
        edge_index = torch.tensor(
            [[e[0] for e in edges], [e[1] for e in edges]],
            dtype=torch.long
        )

        # Convert features and labels
        x = torch.tensor(features.values, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        # Build Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # --- Graph summary ---
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

        print("\n[Graph Summary]")
        print(f"Nodes: {num_nodes}")
        print(f"Edges: {num_edges}")
        print(f"Average Degree: {avg_degree:.2f}")

        return data

"""
Numeric-only feature matrix: (5000, 22), Labels: (5000,)
Auto-encoded feature dtypes:
 FirstName    int16
LastName     int16
Email        int16
City          int8
Race          int8
dtype: object

[Graph Summary]
Nodes: 5000
Edges: 54753
Average Degree: 10.95
Data(x=[5000, 36], edge_index=[2, 54753], y=[5000])

Nodes: 5000 → one per employee.

Edges: 54,753 → massively reduced from ~5.3M earlier ✅.

Average Degree: 10.95 → each employee is connected to ~11 others (manager link + ~k peers).

Data(x=[5000, 36], edge_index=[2, 54753], y=[5000])

5000 employees

36 features each (numeric + encoded categoricals)

54,753 edges

5000 labels



"""