
# -*- coding: utf-8 -*-
# Source: src/modelling/gnn_model_enhanced.py

# -----------------------------------------------------------------------------
# Title: GNN Model for Employee Attrition Prediction with Enhanced Feature Engineering
# Description: This script implements a Graph Neural Network (GNN) model to predict employee attrition.
# It uses the same feature engineering pipeline as the XGBoost and CatBoost models for consistency.
# The model is built using PyTorch Geometric and includes options for different GNN architectures.
# -----------------------------------------------------------------------------

"""Graph Neural Network (GNN) Model for Employee Attrition Prediction with Enhanced Feature Engineering"""

"""

Detailed Explanation of GNN, GCN, GAT, and GraphSAGE in HR Attrition Analysis

Explanation of GNN, GCN, GAT, and GraphSAGE in the Context of HR Attrition Analysis:

GNNs and their variants—Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and GraphSAGE—are specialized ML models designed to handle data structured as graphs, making them suitable for analyzing complex relationships in HR datasets, such as employee attrition influenced by organizational networks.

GNNs operate on graph-structured data, capturing the dependencies between nodes (employees) through their connections (edges). This is particularly useful in HR analytics, where the relationships between employees can significantly impact outcomes like attrition.

How GNNs Work:
1. **Node Representation**: Each employee is represented as a node with features (e.g., tenure, performance scores, engagement levels).
2. **Edge Representation**: Relationships between employees (e.g., reporting lines, team collaborations) are represented as edges.
3. **Message Passing**: GNNs iteratively update node representations by aggregating information from neighboring nodes, allowing the model to learn from both individual features and relational context.

GCNs extend traditional CNNs to graph data by using convolutional layers that aggregate information from a node's neighbors. This allows GCNs to learn representations that consider both the node's features and its local graph structure.
How GCNs Work:
1. **Convolutional Layers**: GCNs apply convolution operations on graphs, where each node's representation is updated based on its neighbors' features.
2. **Layer Stacking**: Multiple GCN layers can be stacked to capture information from further away in the graph, enabling the model to learn more complex relationships.
In context to HR attrition, GCNs can effectively model how an employee's likelihood to leave is influenced by their immediate network, such as their manager and peers. 

GATs introduce attention mechanisms to GNNs, enabling the model to weigh the importance of different neighbors when aggregating information. This is beneficial in HR contexts where not all relationships are equally important for predicting attrition.
How GATs Work:
1. **Attention Mechanism**: GATs compute attention scores for each neighbor, allowing the model to focus on more relevant connections.
2. **Weighted Aggregation**: Node representations are updated by aggregating neighbor features weighted by their attention scores, enhancing the model's ability to capture significant relationships.
In HR attrition analysis, GATs can help identify which relationships (e.g., mentorship, team dynamics) are more influential in an employee's decision to stay or leave.


GraphSAGE takes a different approach by sampling a fixed-size neighborhood around each node, making it scalable to large graphs. This is particularly relevant for HR datasets that may include a vast number of employees and relationships.
How GraphSAGE Works:
1. **Neighborhood Sampling**: GraphSAGE samples a fixed number of neighbors for each node, reducing computational complexity.
2. **Aggregation Functions**: It employs various aggregation functions (mean, LSTM, pooling) to combine information from sampled neighbors, allowing for flexible and efficient learning.   
In HR attrition contexts, GraphSAGE can efficiently handle large organizational structures, capturing essential relational information without being overwhelmed by the graph's size.

Overall, these GNN architectures provide powerful tools for understanding and predicting employee attrition by leveraging the rich relational information present in organizational networks.

"""

"""
How this Code is structured:
1. Imports and Setup: Import necessary libraries and set up paths for local modules.
2. Configuration Management: Define a configuration class to manage hyperparameters and settings for the GNN model.
3. Memory Monitoring Utilities: Implement utilities to monitor and manage memory usage during training.
4. Hybrid GNN Classifier: Define a hybrid GNN architecture that combines graph neural networks with tabular learning.
5. Dense GCN-Optimized Graph Builder: Create a graph builder optimized for GCN with dense organizational connections.
6. Training and Evaluation Functions: Implement functions to train the GNN model and evaluate its performance.
7. Main Execution Block: Set up the main execution flow to load data, build the graph, create the model, and run training and evaluation.
8. Documentation and Comments: Provide detailed comments and documentation throughout the code for clarity.

"""

# -----------------------------
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG imports
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# Add src to path for local imports
script_path = Path(__file__).resolve()
src_path = script_path.parent.parent
sys.path.append(str(src_path))

# Define file paths - match your workspace structure
project_dir = script_path.parent.parent.parent
print(f"Project Directory: {project_dir}")
ENRICHED_FILE_PATH = project_dir / 'data' / 'raw' / 'hrdb_enriched.csv'

# Local modules - updated imports to match your workspace structure
from feature.transform_model_ready_features import FeaturesTransformedModel

# -----------------------------
# Model configurations 
# -----------------------------

"""
Configuration management for GNN models

The GNNConfig class is a configuration management tool that defines and stores hyperparameters and settings for the GNN model training process.

It includes parameters for model architecture, training, data handling, and system settings.

Description of Key Parameters:
- model_type: Type of GNN architecture (e.g., 'gcn', 'gat', 'sage').
- num_layers: Number of GNN layers in the model.
- hidden_channels: Number of hidden units in each GNN layer.
- heads: Number of attention heads (specific to GAT).
- dropout: Dropout rate for regularization.
- use_hybrid: Whether to use a hybrid architecture combining GNN with tabular learning.
- lr: Learning rate for the optimizer.
- weight_decay: Weight decay for regularization.
- epochs: Number of training epochs.
- max_edges: Maximum number of edges in the graph.
- device: Device to run the model on (e.g., 'cuda' or 'cpu').
- seed: Random seed for reproducibility.
- report_dir: Directory to save reports and outputs.
- class_weight_strategy: Strategy for handling class imbalance (e.g., 'balanced_focal').
- top_features_count: Number of top features to use for similarity edges.
- similarity_threshold: Similarity threshold for creating edges based on feature similarity. Controls graph density. More connections with lower values. 
- to_dict/from_dict: Methods to convert the configuration to/from a dictionary for easy saving/loading. 


"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch


@dataclass
class GNNConfig:
    """Configuration class for GNN models"""
    
    # Model architecture
    model_type: str = "gat" # Options: gcn, gat, sage
    num_layers: int = 4 # Minimum 4 for sufficient depth
    hidden_channels: int = 256 # Minimum 256 for sufficient capacity
    heads: int = 8 # GAT-specific parameter - number of attention heads
    dropout: float = 0.3 # Dropout rate for regularization
    use_hybrid: bool = True # Whether to use hybrid GNN + tabular architecture
    
    # Training parameters
    lr: float = 0.002 # Learning rate
    weight_decay: float = 1e-4 # Weight decay for regularization
    epochs: int = 300 # Number of training epochs
    patience: int = 50 # Early stopping patience
    gradient_clip: float = 1.0 # Gradient clipping value
    
    # Data parameters
    max_edges: int = 50000 # Maximum edges in the graph
    test_size: float = 0.2 # Test set proportion
    val_size: float = 0.25 # Validation set proportion
    batch_size: int = 64 # Batch size for training
    # System parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 4 # DataLoader workers
    
    # Paths
    data_path: str = "data/raw/hrdb_enriched.csv" # Path to raw data
    report_dir: str = "reports/advanced_gnn_v3" # Directory for reports and outputs
    
    # Class weighting strategy
    class_weight_strategy: str = "balanced_focal"  # Options: balanced_focal, sqrt_balanced, balanced
    
    # Feature importance
    top_features_count: int = 15 # Number of top features to use for similarity edges
    similarity_threshold: float = 0.8 # Similarity threshold for edge creation

    """
    Convert config to/from dictionary for easy saving/loading
    """
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'model_type': self.model_type,
            'num_layers': self.num_layers,
            'hidden_channels': self.hidden_channels,
            'heads': self.heads,
            'dropout': self.dropout,
            'use_hybrid': self.use_hybrid,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'max_edges': self.max_edges,
            'seed': self.seed
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'GNNConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


# Predefined configurations
QUICK_TEST_CONFIG = GNNConfig(
    epochs=50,
    max_edges=10000,
    hidden_channels=128,
    patience=20
)

FULL_TRAINING_CONFIG = GNNConfig(
    epochs=300,
    max_edges=50000,
    hidden_channels=256,
    patience=50
)

HIGH_PERFORMANCE_CONFIG = GNNConfig(
    epochs=500,
    max_edges=75000,
    hidden_channels=512,
    num_layers=6,
    patience=75,
    lr=0.001
)

# Dense GCN-optimized configuration for 16GB memory systems
DENSE_GCN_CONFIG = GNNConfig(
    model_type="gcn",
    num_layers=4,
    hidden_channels=384,  # Increased capacity for dense graphs
    heads=1,  # GCN doesn't use heads
    dropout=0.4,
    use_hybrid=True,
    
    # Training parameters optimized for dense graphs
    lr=0.001,  # Lower learning rate for stability
    weight_decay=5e-5,  # Reduced regularization
    epochs=400,  # More epochs for dense graph convergence
    patience=60,  # More patience for dense graphs
    gradient_clip=0.5,  # Gradient clipping for stability
    
    # Dense graph parameters
    max_edges=150000,  # Much higher edge capacity
    similarity_threshold=0.5,  # Lower threshold = more connections
    top_features_count=20,  # More features for similarity
    
    # Memory and system
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
    report_dir="reports/dense_gcn_optimized",
    
    # Class weighting for dense graphs
    class_weight_strategy="balanced_focal"
)


# -----------------------------
# Memory monitoring utilities for GNN training
# -----------------------------

"""
What is this section about?
Memory monitoring utilities for GNN training
This section implements utilities to monitor and manage memory usage during the training of GNN models. It includes functions to log memory usage, check available memory, and perform cleanup operations to prevent memory leaks.
Why is it important?
GNN training can be memory-intensive, especially with large graphs and complex architectures. Monitoring memory usage helps ensure that the training process does not exceed system limits, which could lead to crashes or degraded performance. Efficient memory management is crucial for maintaining smooth training workflows and optimizing resource utilization.

"""

import psutil
import torch
import gc
from typing import Optional


class MemoryMonitor:
    """Monitor and manage memory usage during training"""
    
    def __init__(self, log_enabled: bool = True):
        self.log_enabled = log_enabled
        self.peak_memory = 0.0
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        memory_mb = self.process.memory_info().rss / 1024 ** 2
        self.peak_memory = max(self.peak_memory, memory_mb)
        return memory_mb
    
    def get_available_memory(self) -> float:
        """Get available system memory in MB"""
        return psutil.virtual_memory().available / 1024 ** 2
    
    def log_memory(self, stage: str = "") -> float:
        """Log current memory usage"""
        if not self.log_enabled:
            return self.get_memory_usage()
            
        memory_mb = self.get_memory_usage()
        available_mb = self.get_available_memory()
        print(f"[Memory] {stage}: {memory_mb:.1f} MB used, {available_mb:.1f} MB available")
        return memory_mb
    
    def cleanup(self):
        """Force garbage collection and cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage"""
        return self.peak_memory
    
    def check_memory_threshold(self, threshold_mb: float = 14000) -> bool:
        """Check if memory usage is below threshold"""
        current_memory = self.get_memory_usage()
        return current_memory < threshold_mb
    
    def __enter__(self):
        """Context manager entry"""
        self.cleanup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


def monitor_memory(stage: str = "") -> float:
    """Standalone memory monitoring function"""
    monitor = MemoryMonitor()
    return monitor.log_memory(stage)


def cleanup_memory():
    """Standalone memory cleanup function"""
    monitor = MemoryMonitor()
    monitor.cleanup()


###########################
# Hybrid GNN Classifier
###########################

"""
HybridGNNClassifier: A PyTorch nn.Module implementing a hybrid GNN architecture. 
This is the central model class defines a hybrid GNN architecture that combines graph neural networks (GNN) with traditional dense layers for tabular data.
The model captures both relational information from the graph structure and individual feature patterns from tabular data.
Key Features:
- Combines GNNs with dense layers for improved performance
- Captures both graph-based and tabular feature interactions
- Flexible architecture for various data modalities

In the context of employee attrition prediction, this hybrid approach allows the model to leverage both the organizational relationships (e.g., reporting lines, team structures) and individual employee features (e.g., tenure, performance scores) to make more accurate predictions.

This class can be instantiated with different GNN architectures (GCN, GAT, GraphSAGE) and includes options for dropout, attention mechanisms, and feature importance weighting.
The architecture consists of:
-   A GNN branch to capture organizational relationships (e.g., manager-employee, department peers, project collaborations) using graph convolutional layers (GCN, GAT, or GraphSAGE).
-   A tabular branch to process individual employee features (e.g., tenure, performance rating, job satisfaction). Useful when individual features are strong predictors.
-   A fusion layer to combine representations from both branches for final attrition predictions. Useful when both relational and individual features are important.
-   An attention mechanism to weigh the importance of input features. Useful for interpretability and understanding feature contributions.



Flow of the code:
1. Initialization (__init__):
   - Sets up the GNN layers based on the specified architecture (GCN, GAT, GraphSAGE).
   - Initializes dense layers for tabular data processing.
   - Configures dropout and attention mechanisms for feature importance.
2. Forward Pass (forward):
   - Implements the forward pass for both GNN and tabular branches.
   - Combines the outputs from both branches using a fusion layer.
3. Embedding Extraction (get_embeddings):
    - Provides a method to extract node embeddings from the GNN branch for further analysis or visualization.
    - This can be useful for tasks such as clustering, visualization, or as input to other models.
4. Parameter Counting (count_parameters):
   - Implements a method to count the total number of trainable parameters in the model.
   - This can be useful for model analysis and debugging.

   
Key Methods:
- __init__: Initializes the model architecture, including GNN layers, dense layers, and attention mechanisms.
- forward: Defines the forward pass through the model, combining GNN and tabular branches. 
- get_embeddings: Extracts node embeddings from the GNN branch.
- count_parameters: Counts the total number of trainable parameters in the model.
- reset_parameters: Resets the model parameters to their initial values.


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
from typing import Optional


class HybridGNNClassifier(nn.Module):
    """
    Hybrid architecture combining GNN with tabular learning
    - GNN captures organizational relationships
    - Dense layers capture individual feature patterns
    - Fusion layer combines both representations
    """
    
    def __init__(self, conv_type: str, in_channels: int, hidden_channels: int, 
                 out_channels: int, num_layers: int = 4, dropout: float = 0.3, 
                 heads: int = 8, use_tabular_branch: bool = True):
        """
        - Initialize hybrid GNN classifier
        Args:
            conv_type: Type of GNN layer ('gcn', 'gat', 'sage')
            in_channels: Number of input features
            hidden_channels: Number of hidden units in GNN layers
            out_channels: Number of output classes (e.g., 2 for binary classification)
            num_layers: Number of GNN layers
            dropout: Dropout rate for regularization
            heads: Number of attention heads (for GAT)
            use_tabular_branch: Whether to include the tabular branch   
        """
        # Sets up the GNN branch, tabular branch, fusion layer, and attention mechanism
        super().__init__()
        self.conv_type = conv_type.lower()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_tabular_branch = use_tabular_branch
        
        # Ensure minimum capacity
        hidden_channels = max(hidden_channels, 256)
        
        # Initialize components
        self._init_gnn_branch(in_channels, hidden_channels, heads)
        self._init_tabular_branch(in_channels, hidden_channels)
        self._init_fusion_layer(hidden_channels, out_channels)
        self._init_attention_mechanism(in_channels)
        
        self.reset_parameters()
    
    def _init_gnn_branch(self, in_channels: int, hidden_channels: int, heads: int):
        """
        - Initialize GNN branch components
        - Supports GCN, GAT, and GraphSAGE architectures
        - Uses multiple layers with batch normalization and dropout
        - Input projection layer to map input features to hidden dimension
        - Output projection layer to reduce hidden dimension before fusion
        - Residual connections for stable training
        - Attention mechanism to weigh feature importance
        - Dropout for regularization
        - Layer normalization for stable training
        """

        self.gnn_convs = nn.ModuleList()
        self.gnn_bns = nn.ModuleList()
        
        # Input projection
        self.gnn_input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout * 0.3)
        )
        
        # GNN layers
        for i in range(self.num_layers):
            self.gnn_convs.append(self._make_conv(self.conv_type, hidden_channels, hidden_channels, heads))
            self.gnn_bns.append(BatchNorm(hidden_channels))
        
        # Output projection
        self.gnn_output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
    
    def _init_tabular_branch(self, in_channels: int, hidden_channels: int):
        """
        - Initialize tabular branch components
        - Dense layers to process individual features
        - Layer normalization and dropout for regularization
        - Reduces dimensionality before fusion
        - Useful when individual features are strong predictors
        
        """
        if self.use_tabular_branch:
            self.tabular_layers = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.LayerNorm(hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                
                nn.Linear(hidden_channels // 2, hidden_channels // 4),
                nn.LayerNorm(hidden_channels // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout * 0.5)
            )
    
    def _init_fusion_layer(self, hidden_channels: int, out_channels: int):
        """
        - Initialize fusion layer
        - Combines GNN and tabular representations
        - Dense layers with dropout and layer normalization
        - Outputs final class predictions
        - Useful when both relational and individual features are important

        """
        if self.use_tabular_branch:
            fusion_input = (hidden_channels // 2) + (hidden_channels // 4)  # GNN + Tabular
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input, hidden_channels // 2),
                nn.LayerNorm(hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                
                nn.Linear(hidden_channels // 2, hidden_channels // 4),
                nn.LayerNorm(hidden_channels // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout * 0.5),
                
                nn.Linear(hidden_channels // 4, out_channels)
            )
        else:
            # GNN-only classifier
            self.fusion = nn.Sequential(
                nn.Linear(hidden_channels // 2, hidden_channels // 4),
                nn.LayerNorm(hidden_channels // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_channels // 4, out_channels)
            )
    
    def _init_attention_mechanism(self, in_channels: int):
        """
        - Initialize attention mechanism for feature importance
        - Weighs input features to focus on important ones
        - Uses a small feedforward network with sigmoid activation
        - Useful for interpretability and understanding feature contributions

        """

        self.feature_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
    
    def _make_conv(self, conv_type: str, in_ch: int, out_ch: int, heads: int):
        """
        - Create convolution layer based on type
        - Supports GCN, GAT, and GraphSAGE
        - Configures layer parameters for stability and performance        
        """
        if conv_type == "gcn":
            return GCNConv(in_ch, out_ch, normalize=True, add_self_loops=True, bias=True)
        elif conv_type == "gat":
            return GATConv(in_ch, out_ch, heads=heads, concat=False, dropout=0.1,
                          add_self_loops=True, bias=True)
        elif conv_type == "sage":
            return SAGEConv(in_ch, out_ch, normalize=True, bias=True)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
    
    def reset_parameters(self):
        """
        Initialize parameters with advanced strategies
        - Xavier initialization for linear layers
        """
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if m.out_features == 2:  # Final classification layer
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.apply(init_weights)
        
        # Reset conv layer parameters
        for conv in self.gnn_convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        -  Forward pass through hybrid architecture
        -  Combines GNN and tabular features
        -  Applies feature attention
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            batch: Batch vector for mini-batch training (optional)
            edge_weight: Edge weights for weighted graphs (optional)
        Returns:
            output: Class logits [num_nodes, out_channels]

        
        """
        # Feature attention
        att_weights = self.feature_attention(x)
        x_attended = x * att_weights
        
        # GNN Branch
        gnn_out = self._forward_gnn_branch(x_attended, edge_index, edge_weight)
        
        # Tabular Branch (if enabled)
        if self.use_tabular_branch:
            tab_out = self.tabular_layers(x_attended)
            combined = torch.cat([gnn_out, tab_out], dim=1)
            output = self.fusion(combined)
        else:
            output = self.fusion(gnn_out)
        
        return output
    
    def _forward_gnn_branch(self, x: torch.Tensor, edge_index: torch.Tensor, 
                           edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """
        - Forward pass through GNN branch
        - Applies multiple GNN layers with residual connections
        - Uses batch normalization, ReLU, and dropout
        - Residual connections for stable training; dropout for regularization
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            edge_weight: Edge weights for weighted graphs (optional)
        Returns:
            output: Node embeddings [num_nodes, hidden_channels]
        """
        gnn_x = self.gnn_input_proj(x)
        
        for i, (conv, bn) in enumerate(zip(self.gnn_convs, self.gnn_bns)):
            residual = gnn_x
            
            # Graph convolution
            if self.conv_type == "gcn" and edge_weight is not None:
                gnn_x = conv(gnn_x, edge_index, edge_weight)
            else:
                gnn_x = conv(gnn_x, edge_index)
            
            gnn_x = bn(gnn_x)
            gnn_x = F.relu(gnn_x)
            
            # Residual connection
            if gnn_x.shape == residual.shape:
                gnn_x = gnn_x + residual
            
            gnn_x = F.dropout(gnn_x, p=self.dropout, training=self.training)
        
        return self.gnn_output(gnn_x)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, 
                      edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        - Get node embeddings from GNN branch
        - Applies feature attention before GNN processing
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            edge_weight: Edge weights for weighted graphs (optional)
        Returns:
            output: Node embeddings [num_nodes, hidden_channels]
        """
        att_weights = self.feature_attention(x)
        x_attended = x * att_weights
        return self._forward_gnn_branch(x_attended, edge_index, edge_weight)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config, in_channels: int) -> HybridGNNClassifier:
    """Factory function to create model from config"""
    return HybridGNNClassifier(
        conv_type=config.model_type,
        in_channels=in_channels,
        hidden_channels=config.hidden_channels,
        out_channels=2,
        num_layers=config.num_layers,
        dropout=config.dropout,
        heads=config.heads,
        use_tabular_branch=config.use_hybrid
    )

#############################
# Dense GCN-Optimized Graph Builder
#############################

"""
The DenseGCNGraphBuilder class constructs a dense, weighted graph of employee data with enhanced organizational relationships,
 optimized feature engineering, and multiple edge types, enabling effective GCN-based prediction of HR attrition by capturing complex workplace interactions.

Graph Representation:
    - Nodes represent employees.
    - Edges represent various organizational relationships (e.g., hierarchy, project collaboration, skill overlap).
    - Edge weights reflect the strength of relationships.
    - Node features are engineered HR attributes (e.g., tenure, performance rating).

Key Features:
    - Creates a highly connected graph (density >1%) to capture complex organizational patterns, such as team cohesion or cross-department interactions, which impact attrition.
    - Creates multiple Edge Types including hierarchy, similarity, department, performance, project, event, skill, mentorship, and weak tie edges, providing a comprehensive view of factors influencing turnover.
    - Incorporates domain knowledge to define meaningful relationships, enhancing the model's ability to learn relevant patterns.
    - Uses cosine similarity and Euclidean distance to create edges based on feature similarity, capturing nuanced relationships.
    - Implements a lower similarity threshold (0.5) to create more edges, capturing subtle relationships.
    - Integrates synthetic organizational data (e.g., project assignments, skill overlaps) to enrich connections.
    - Optimizes feature engineering for memory efficiency, handling categorical variables and scaling.
    - Uses Random Forest to select top predictive features (e.g., performance rating, years at company) for similarity-based edges, ensuring relevance to attrition.
    - Handles large graphs with up to 150,000 edges, ensuring scalability for real-world HR datasets.
    - Monitors memory usage to ensure the graph fits within system limits (e.g., 16GB RAM).
    - Provides detailed logging of graph statistics and memory usage.

Flow of the code:
1. Initialization (__init__):
   - Sets up parameters for graph construction, including maximum edges, similarity thresholds, and feature selection.
   - Applies feature engineering to prepare node features.
   - Learns feature importance using Random Forest to select top features for similarity edges.
   - Generates synthetic organizational data to enrich the graph.
2. Feature Preparation (_prepare_features):
   - Applies feature engineering to the raw employee data. 
   - Removes leakage features to prevent data contamination.
3. Feature Optimization (_optimize_features):
    - Optimizes features for memory efficiency, handling categorical variables and scaling.
    - Converts categorical variables to one-hot encoding or label encoding based on cardinality.
    - Ensures all features are numeric and of appropriate data types.
    - Scales features using RobustScaler to handle outliers.
4. Feature Importance Learning (_learn_feature_importance):
   - Uses Random Forest to identify top features predictive of attrition.
   - Features like tenure, performance rating, or job satisfaction are prioritized for similarity-based edge construction.
   - Uses feature importance scores to refine edge selection.
5. Organizational Enrichment (_generate_organizational_enrichment):
   - Creates synthetic organizational relationships (e.g., project assignments, skill overlaps) to enrich the graph.
   - Simulates employee participation in 50 projects (binary matrix)
   - Simulates skill overlaps among employees (binary matrix)
   - Creates weak ties based on co-occurrence in projects or skills
   - Adds edges based on department and performance similarity
   - Adds mentorship edges based on hierarchical relationships
6. Graph Construction (build_graph):
   - Constructs the graph using the prepared node features and edges.
   - Ensures the graph is undirected and includes self-loops.
   - Limits the number of edges to the specified maximum. 
7. Memory Monitoring:
   - Monitors memory usage throughout the graph construction process.
   - Logs memory usage at key stages to identify potential bottlenecks.
   - Implements garbage collection to free up memory as needed.

"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import gc
from typing import List, Tuple, Dict
from pathlib import Path
import sys

# Add src to path for local imports
script_path = Path(__file__).resolve()
src_path = script_path.parent.parent.parent
sys.path.append(str(src_path))

from feature.transform_model_ready_features import FeaturesTransformedModel



class DenseGCNGraphBuilder:
    """
    Dense graph builder optimized for GCN with comprehensive organizational relationships
    """
    
    def __init__(self, df_raw: pd.DataFrame, max_edges: int = 150000, 
                 similarity_threshold: float = 0.5, top_features_count: int = 20):
        """
        Initialize dense graph builder with relaxed constraints
        
        Args:
            df_raw: Raw employee dataframe
            max_edges: Maximum edges (increased for dense graph)
            similarity_threshold: Lower threshold for more connections (0.5 vs 0.8)
            top_features_count: More features for similarity (20 vs 15)
        """
        self.df_raw = df_raw.copy()
        self.max_edges = max_edges
        self.similarity_threshold = similarity_threshold  # Lower = more connections
        self.top_features_count = top_features_count
        self.memory_monitor = MemoryMonitor()
        
        print("[Info] Building DENSE GCN-optimized graph with enhanced organizational semantics...")
        print(f"[Config] Max Edges: {max_edges:,}, Similarity Threshold: {similarity_threshold}")
        self.memory_monitor.log_memory("Dense Graph Builder Init")
        
        # Apply feature engineering
        self._prepare_features()
        self.node_ids = {eid: idx for idx, eid in enumerate(self.df_raw['EmployeeID'])}
        
        # Learn feature importance for edge weighting
        self._learn_feature_importance()
        
        # Generate synthetic organizational data for richer connections
        self._generate_organizational_enrichment()
        
        print(f"[Info] Graph features shape: {self.X_engineered.shape}")
        print(f"[Info] Target distribution: {pd.Series(self.y).value_counts().to_dict()}")
        
        self.memory_monitor.log_memory("After Dense Feature Engineering")
    
    def _prepare_features(self):
        """Prepare and optimize features"""
        fe = FeaturesTransformedModel(self.df_raw, target_col="Attrition")
        self.X_engineered, self.y = fe.prepare_features(return_df=True)
        
        # Remove leakage features
        leakage_cols = ["attrition_prob", "Attrition"]
        for col in leakage_cols:
            if col in self.X_engineered.columns:
                self.X_engineered = self.X_engineered.drop(columns=[col])
                print(f"[Info] Removed leakage feature: {col}")
        
        self._optimize_features()
    
    def _optimize_features(self):
        """Memory-efficient feature optimization with robust data type handling"""
        print("[Info] Dense feature optimization...")
        
        # Convert categoricals efficiently
        object_cols = self.X_engineered.select_dtypes(include=['object']).columns
        category_cols = self.X_engineered.select_dtypes(include=['category']).columns
        
        for col in list(object_cols) + list(category_cols):
            if self.X_engineered[col].nunique() <= 25:  # Increased threshold
                # One-hot encode with proper handling
                try:
                    col_data = self.X_engineered[col].astype(str)
                    dummies = pd.get_dummies(col_data, prefix=col, dummy_na=True)
                    
                    for dummy_col in dummies.columns:
                        dummies[dummy_col] = dummies[dummy_col].astype(np.float32)
                    
                    self.X_engineered = self.X_engineered.drop(columns=[col])
                    self.X_engineered = pd.concat([self.X_engineered, dummies], axis=1)
                    
                except Exception as e:
                    print(f"[Warning] One-hot encoding failed for {col}: {e}")
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    self.X_engineered[col] = le.fit_transform(self.X_engineered[col].astype(str))
            else:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                self.X_engineered[col] = le.fit_transform(self.X_engineered[col].astype(str))
        
        # Convert all columns to numeric
        for col in self.X_engineered.columns:
            try:
                if not pd.api.types.is_numeric_dtype(self.X_engineered[col]):
                    self.X_engineered[col] = pd.to_numeric(self.X_engineered[col], errors='coerce')
                
                if pd.api.types.is_numeric_dtype(self.X_engineered[col]):
                    self.X_engineered[col] = self.X_engineered[col].astype(np.float32)
                else:
                    self.X_engineered[col] = pd.to_numeric(
                        self.X_engineered[col].astype(str), errors='coerce'
                    ).fillna(0).astype(np.float32)
                    
            except Exception as e:
                print(f"[Warning] Dropping problematic column {col}: {e}")
                self.X_engineered = self.X_engineered.drop(columns=[col])
        
        # Fill NaNs and apply scaling
        self.X_engineered = self.X_engineered.fillna(0)
        
        print("[Info] Applying robust scaling...")
        scaler = RobustScaler()
        
        try:
            scaled_data = scaler.fit_transform(self.X_engineered.values)
            self.X_scaled = pd.DataFrame(
                scaled_data,
                columns=self.X_engineered.columns,
                index=self.X_engineered.index
            )
        except Exception as e:
            print(f"[Warning] Scaling failed: {e}, using original data")
            self.X_scaled = self.X_engineered.copy()
        
        print(f"[Info] Final feature matrix shape: {self.X_engineered.shape}")
        gc.collect()
    
    def _learn_feature_importance(self):
        """Learn feature importance with more comprehensive analysis"""
        print("[Info] Learning feature importance for intelligent edge construction...")
        
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)  # More trees
        sample_size = min(3000, len(self.X_engineered))  # Larger sample
        rf.fit(
            self.X_engineered.sample(n=sample_size, random_state=42), 
            pd.Series(self.y).iloc[:sample_size]
        )
        
        self.feature_importance = dict(zip(self.X_engineered.columns, rf.feature_importances_))
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        self.top_features = [f for f, _ in sorted_features[:self.top_features_count]]
        
        print(f"[Info] Top {len(self.top_features)} predictive features: {self.top_features[:5]}")
        self.memory_monitor.log_memory("After Enhanced Feature Importance Learning")
    
    def _generate_organizational_enrichment(self):
        """Generate synthetic organizational data for richer connections"""
        print("[Info] Generating organizational enrichment data...")
        
        n_employees = len(self.df_raw)
        np.random.seed(42)
        
        # 1. Project history simulation
        n_projects = 50
        project_participation = np.random.binomial(1, 0.3, (n_employees, n_projects))
        self.project_matrix = pd.DataFrame(
            project_participation, 
            index=self.df_raw['EmployeeID'], 
            columns=[f'Project_{i}' for i in range(n_projects)]
        )
        
        # 2. Company-wide events simulation
        events = ['Townhall_Q1', 'Townhall_Q2', 'Townhall_Q3', 'Townhall_Q4',
                 'Holiday_Party', 'Summer_Retreat', 'Training_Day', 'All_Hands',
                 'Innovation_Day', 'Diversity_Event']
        event_attendance = np.random.binomial(1, 0.6, (n_employees, len(events)))
        self.event_matrix = pd.DataFrame(
            event_attendance,
            index=self.df_raw['EmployeeID'],
            columns=events
        )
        
        # 3. Skill overlap simulation
        skills = ['Python', 'SQL', 'Leadership', 'Analytics', 'Communication',
                 'Project_Mgmt', 'Strategy', 'Finance', 'Marketing', 'Sales']
        skill_levels = np.random.randint(0, 4, (n_employees, len(skills)))  # 0-3 skill level
        self.skill_matrix = pd.DataFrame(
            skill_levels,
            index=self.df_raw['EmployeeID'],
            columns=skills
        )
        
        # 4. Mentorship networks
        self.mentorship_pairs = []
        senior_employees = self.df_raw[self.df_raw['YearsAtCompany'] >= 5]['EmployeeID'].tolist()
        junior_employees = self.df_raw[self.df_raw['YearsAtCompany'] <= 2]['EmployeeID'].tolist()
        
        for junior in junior_employees[:200]:  # First 200 junior employees
            if len(senior_employees) > 0:
                mentor = np.random.choice(senior_employees)
                self.mentorship_pairs.append((mentor, junior))
        
        print(f"[Info] Generated organizational data:")
        print(f"  - Project matrix: {self.project_matrix.shape}")
        print(f"  - Event attendance: {self.event_matrix.shape}")  
        print(f"  - Skill profiles: {self.skill_matrix.shape}")
        print(f"  - Mentorship pairs: {len(self.mentorship_pairs)}")
    
    def build_dense_graph(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build sophisticated dense multi-layer graph structure"""
        print(f"[Info] Building DENSE graph (target: {self.max_edges:,} edges)...")
        
        all_edges, all_weights, all_types = [], [], []
        
        # Enhanced edge builders with dense connections
        edge_builders = [
            ("hierarchy", self._build_enhanced_hierarchy_edges),
            ("dense_similarity", self._build_dense_similarity_edges),
            ("enhanced_department", self._build_enhanced_department_networks),
            ("performance_collaboration", self._build_performance_collaboration_edges),
            ("project_history", self._build_project_history_edges),
            ("company_events", self._build_company_event_edges),
            ("skill_overlap", self._build_skill_overlap_edges),
            ("mentorship", self._build_mentorship_edges),
            ("cross_department_events", self._build_cross_department_event_edges),
            ("weak_ties", self._build_weak_tie_edges)
        ]
        
        for edge_type, builder_func in edge_builders:
            try:
                edges, weights = builder_func()
                all_edges.extend(edges)
                all_weights.extend(weights)
                all_types.extend([edge_type] * len(edges))
                print(f"[Info] Added {len(edges):,} {edge_type} edges")
            except Exception as e:
                print(f"[Warning] Failed to build {edge_type} edges: {e}")
        
        # Intelligent filtering for density
        final_edges, final_weights = self._intelligent_dense_filtering(
            all_edges, all_weights, all_types
        )
        
        self._print_dense_graph_statistics(final_edges)
        return final_edges, final_weights
    
    def _build_enhanced_hierarchy_edges(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Enhanced hierarchical edges with skip levels and peer managers"""
        edges, weights = [], []
        
        if 'EmployeeID' not in self.df_raw.columns or 'CurrentManager' not in self.df_raw.columns:
            return edges, weights
        
        # Direct manager relationships
        for _, row in self.df_raw.dropna(subset=['CurrentManager']).iterrows():
            manager_id = row['CurrentManager']
            emp_id = row['EmployeeID']
            
            if manager_id in self.node_ids and emp_id in self.node_ids:
                manager_idx = self.node_ids[manager_id]
                emp_idx = self.node_ids[emp_id]
                
                edges.extend([(manager_idx, emp_idx), (emp_idx, manager_idx)])
                weights.extend([1.0, 0.85])  # Slightly higher weights
        
        # Skip-level and peer manager connections
        manager_map = dict(zip(self.df_raw['EmployeeID'], self.df_raw['CurrentManager']))
        
        # Skip-level (grandmanager connections) - simplified
        for emp_id, manager_id in manager_map.items():
            if pd.isna(manager_id):
                continue
                
            grandmanager_id = manager_map.get(manager_id)
            if grandmanager_id and not pd.isna(grandmanager_id):
                if emp_id in self.node_ids and grandmanager_id in self.node_ids:
                    emp_idx = self.node_ids[emp_id]
                    gm_idx = self.node_ids[grandmanager_id]
                    edges.extend([(gm_idx, emp_idx), (emp_idx, gm_idx)])
                    weights.extend([0.7, 0.5])
        
        # Peer managers (managers at same level) - simplified approach
        manager_levels = {}
        for emp_id, manager_id in manager_map.items():
            if not pd.isna(manager_id) and manager_id in manager_map:
                level = 1
                current = manager_id
                # Limit depth to prevent infinite loops
                for _ in range(5):  # Max 5 levels
                    if current not in manager_map or pd.isna(manager_map.get(current)):
                        break
                    current = manager_map[current]
                    level += 1
                manager_levels[emp_id] = level
        
        # Connect managers at same level
        level_groups = {}
        for emp_id, level in manager_levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(emp_id)
        
        for level, managers in level_groups.items():
            if len(managers) > 1:
                for i, mgr1 in enumerate(managers):
                    for mgr2 in managers[i+1:min(i+6, len(managers))]:  # Connect to 5 peers
                        if mgr1 in self.node_ids and mgr2 in self.node_ids:
                            idx1, idx2 = self.node_ids[mgr1], self.node_ids[mgr2]
                            edges.extend([(idx1, idx2), (idx2, idx1)])
                            weights.extend([0.6, 0.6])
        
        return edges, weights
    
    def _build_dense_similarity_edges(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build dense similarity edges with lower threshold"""
        edges, weights = [], []
        
        top_feature_data = self.X_scaled[self.top_features].values
        print(f"[Info] Computing DENSE similarity using {len(self.top_features)} features...")
        print(f"[Info] Similarity threshold: {self.similarity_threshold} (lower = more connections)")
        
        # Use larger chunks for better memory utilization
        chunk_size = 500
        n_samples = len(top_feature_data)
        
        for i in range(0, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)
            chunk_sim = cosine_similarity(top_feature_data[i:end_i], top_feature_data)
            
            for local_i, global_i in enumerate(range(i, end_i)):
                similarities = chunk_sim[local_i]
                
                # Use lower threshold and more connections
                top_k = min(50, n_samples - 1)  # Connect to more peers
                top_indices = np.argsort(similarities)[-top_k-1:-1]
                
                for j in top_indices:
                    sim_score = similarities[j]
                    if sim_score > self.similarity_threshold and global_i != j:
                        edges.append((global_i, j))
                        # Higher weight for very similar employees
                        weight = min(sim_score * 1.1, 1.0)
                        weights.append(weight)
        
        return edges, weights
    
    def _build_enhanced_department_networks(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Enhanced department networks with more cross-connections"""
        edges, weights = [], []
        
        if 'Department' not in self.df_raw.columns:
            return edges, weights
        
        dept_groups = self.df_raw.groupby('Department')
        
        for dept, group in dept_groups:
            if len(group) <= 2:
                continue
            
            emp_ids = group['EmployeeID'].tolist()
            
            # Same role connections (increased density)
            if 'JobRole' in self.df_raw.columns:
                role_groups = group.groupby('JobRole')
                for role, role_group in role_groups:
                    role_emp_ids = role_group['EmployeeID'].tolist()
                    
                    # Connect more employees within same role
                    max_same_role = min(len(role_emp_ids), 40)
                    for i in range(max_same_role):
                        for j in range(i+1, min(i+12, len(role_emp_ids))):  # More connections
                            emp1, emp2 = role_emp_ids[i], role_emp_ids[j]
                            if emp1 in self.node_ids and emp2 in self.node_ids:
                                idx1, idx2 = self.node_ids[emp1], self.node_ids[emp2]
                                edges.extend([(idx1, idx2), (idx2, idx1)])
                                weights.extend([0.85, 0.85])
            
            # Cross-role connections (much more dense)
            max_cross_role = min(len(emp_ids), 60)  # Increased from 30
            for i in range(max_cross_role):
                for j in range(i+1, min(i+15, len(emp_ids))):  # More cross-connections
                    emp1, emp2 = emp_ids[i], emp_ids[j]
                    if emp1 in self.node_ids and emp2 in self.node_ids:
                        idx1, idx2 = self.node_ids[emp1], self.node_ids[emp2]
                        edges.extend([(idx1, idx2), (idx2, idx1)])
                        weights.extend([0.6, 0.6])
        
        return edges, weights
    
    def _build_performance_collaboration_edges(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Enhanced performance-based edges with weaker ties"""
        edges, weights = [], []
        
        if 'PerformanceRating' not in self.df_raw.columns:
            return edges, weights
        
        # High performers (rating 4-5)
        high_perf = self.df_raw[self.df_raw['PerformanceRating'] >= 4]
        if len(high_perf) > 1:
            emp_ids = high_perf['EmployeeID'].tolist()
            max_connections = min(len(emp_ids), 80)  # More connections
            
            for i in range(max_connections):
                for j in range(i+1, min(i+15, len(emp_ids))):  # More per employee
                    emp1, emp2 = emp_ids[i], emp_ids[j]
                    if emp1 in self.node_ids and emp2 in self.node_ids:
                        idx1, idx2 = self.node_ids[emp1], self.node_ids[emp2]
                        edges.extend([(idx1, idx2), (idx2, idx1)])
                        rating1 = high_perf[high_perf['EmployeeID'] == emp1]['PerformanceRating'].iloc[0]
                        rating2 = high_perf[high_perf['EmployeeID'] == emp2]['PerformanceRating'].iloc[0]
                        weight = 0.7 + 0.1 * (rating1 + rating2 - 6)  # Weight by performance
                        weights.extend([weight, weight])
        
        # Medium performers (rating 3) - weaker ties
        med_perf = self.df_raw[self.df_raw['PerformanceRating'] == 3]
        if len(med_perf) > 1:
            emp_ids = med_perf['EmployeeID'].tolist()
            max_connections = min(len(emp_ids), 60)
            
            for i in range(max_connections):
                for j in range(i+1, min(i+8, len(emp_ids))):
                    emp1, emp2 = emp_ids[i], emp_ids[j]
                    if emp1 in self.node_ids and emp2 in self.node_ids:
                        idx1, idx2 = self.node_ids[emp1], self.node_ids[emp2]
                        edges.extend([(idx1, idx2), (idx2, idx1)])
                        weights.extend([0.5, 0.5])  # Weaker ties
        
        return edges, weights
    
    def _build_project_history_edges(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build edges based on project collaboration history"""
        edges, weights = [], []
        
        project_data = self.project_matrix
        
        # Find employees who worked on same projects
        for project in project_data.columns:
            project_members = project_data[project_data[project] == 1].index.tolist()
            
            if len(project_members) >= 2:
                # Connect all project members
                for i, emp1 in enumerate(project_members):
                    for emp2 in project_members[i+1:min(i+10, len(project_members))]:
                        if emp1 in self.node_ids and emp2 in self.node_ids:
                            idx1, idx2 = self.node_ids[emp1], self.node_ids[emp2]
                            edges.extend([(idx1, idx2), (idx2, idx1)])
                            
                            # Weight by project team size (smaller = stronger bond)
                            team_size = len(project_members)
                            weight = max(0.3, 1.0 - (team_size - 2) * 0.05)
                            weights.extend([weight, weight])
        
        return edges, weights
    
    def _build_company_event_edges(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build edges based on company event attendance"""
        edges, weights = [], []
        
        event_data = self.event_matrix
        
        for event in event_data.columns:
            attendees = event_data[event_data[event] == 1].index.tolist()
            
            if len(attendees) >= 5:  # Only events with meaningful attendance
                # Connect random subset of attendees
                max_connections = min(len(attendees), 100)
                selected_attendees = np.random.choice(attendees, max_connections, replace=False)
                
                for i, emp1 in enumerate(selected_attendees):
                    for emp2 in selected_attendees[i+1:min(i+8, len(selected_attendees))]:
                        if emp1 in self.node_ids and emp2 in self.node_ids:
                            idx1, idx2 = self.node_ids[emp1], self.node_ids[emp2]
                            edges.extend([(idx1, idx2), (idx2, idx1)])
                            
                            # Different weights for different event types
                            if 'Townhall' in event:
                                weight = 0.4
                            elif 'Training' in event:
                                weight = 0.6
                            else:
                                weight = 0.5
                            weights.extend([weight, weight])
        
        return edges, weights
    
    def _build_skill_overlap_edges(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build edges based on skill similarity"""
        edges, weights = [], []
        
        skill_data = self.skill_matrix.values
        n_employees = len(skill_data)
        
        # Compute skill similarity (cosine similarity)
        chunk_size = 200
        
        for i in range(0, n_employees, chunk_size):
            end_i = min(i + chunk_size, n_employees)
            chunk_sim = cosine_similarity(skill_data[i:end_i], skill_data)
            
            for local_i, global_i in enumerate(range(i, end_i)):
                similarities = chunk_sim[local_i]
                
                # Find employees with similar skills
                similar_indices = np.where(similarities > 0.6)[0]  # 60% skill overlap
                
                for j in similar_indices:
                    if global_i != j:
                        emp1 = self.skill_matrix.index[global_i]
                        emp2 = self.skill_matrix.index[j]
                        
                        if emp1 in self.node_ids and emp2 in self.node_ids:
                            idx1, idx2 = self.node_ids[emp1], self.node_ids[emp2]
                            edges.append((idx1, idx2))
                            weights.append(similarities[j] * 0.7)
        
        return edges, weights
    
    def _build_mentorship_edges(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build mentorship relationship edges"""
        edges, weights = [], []
        
        for mentor_id, mentee_id in self.mentorship_pairs:
            if mentor_id in self.node_ids and mentee_id in self.node_ids:
                mentor_idx = self.node_ids[mentor_id]
                mentee_idx = self.node_ids[mentee_id]
                
                # Bidirectional but weighted differently
                edges.extend([(mentor_idx, mentee_idx), (mentee_idx, mentor_idx)])
                weights.extend([0.9, 0.7])  # Mentor -> mentee stronger
        
        return edges, weights
    
    def _build_cross_department_event_edges(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build cross-department edges through company events"""
        edges, weights = [], []
        
        if 'Department' not in self.df_raw.columns:
            return edges, weights
        
        # Group employees by department
        dept_groups = self.df_raw.groupby('Department')['EmployeeID'].apply(list).to_dict()
        
        # For each major company event, connect people across departments
        major_events = ['Townhall_Q1', 'Townhall_Q2', 'All_Hands', 'Innovation_Day']
        
        for event in major_events:
            if event in self.event_matrix.columns:
                attendees = self.event_matrix[self.event_matrix[event] == 1].index.tolist()
                
                # Group attendees by department
                dept_attendees = {}
                for emp_id in attendees:
                    if emp_id in self.df_raw['EmployeeID'].values:
                        dept = self.df_raw[self.df_raw['EmployeeID'] == emp_id]['Department'].iloc[0]
                        if dept not in dept_attendees:
                            dept_attendees[dept] = []
                        dept_attendees[dept].append(emp_id)
                
                # Connect across departments
                dept_list = list(dept_attendees.keys())
                for i, dept1 in enumerate(dept_list):
                    for dept2 in dept_list[i+1:]:
                        # Connect subset of employees across departments
                        emp1_list = dept_attendees[dept1][:10]  # Max 10 per dept
                        emp2_list = dept_attendees[dept2][:10]
                        
                        for emp1 in emp1_list:
                            for emp2 in emp2_list[:3]:  # Max 3 cross-connections per person
                                if emp1 in self.node_ids and emp2 in self.node_ids:
                                    idx1, idx2 = self.node_ids[emp1], self.node_ids[emp2]
                                    edges.extend([(idx1, idx2), (idx2, idx1)])
                                    weights.extend([0.4, 0.4])  # Weaker cross-dept ties
        
        return edges, weights
    
    def _build_weak_tie_edges(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Build weak tie edges for social network completeness"""
        edges, weights = [], []
        
        # Random weak ties to simulate casual interactions
        n_employees = len(self.df_raw)
        emp_indices = list(range(n_employees))
        
        # Add random weak ties (small world property)
        n_weak_ties = min(5000, n_employees * 2)
        
        for _ in range(n_weak_ties):
            emp1, emp2 = np.random.choice(emp_indices, 2, replace=False)
            
            # Weak tie with low weight
            edges.append((emp1, emp2))
            weights.append(0.2)
        
        return edges, weights
    
    def _intelligent_dense_filtering(self, edges: List[Tuple[int, int]], 
                                   weights: List[float], types: List[str]) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Filter edges intelligently for dense graph"""
        print(f"[Info] Dense filtering from {len(edges):,} edges to {self.max_edges:,}...")
        
        edge_data = list(zip(edges, weights, types))
        
        # Remove duplicates, keeping highest weight
        edge_dict = {}
        for (src, dst), weight, edge_type in edge_data:
            edge_key = (min(src, dst), max(src, dst))
            if edge_key not in edge_dict or edge_dict[edge_key][0] < weight:
                edge_dict[edge_key] = (weight, edge_type, src, dst)
        
        # Priority for dense graph
        type_priority = {
            'hierarchy': 10, 'mentorship': 9, 'dense_similarity': 8,
            'performance_collaboration': 7, 'project_history': 6,
            'enhanced_department': 5, 'company_events': 4,
            'skill_overlap': 3, 'cross_department_events': 2, 'weak_ties': 1
        }
        
        def edge_importance(item):
            weight, edge_type, _, _ = item[1]
            return weight * type_priority.get(edge_type, 1)
        
        sorted_edges = sorted(edge_dict.items(), key=edge_importance, reverse=True)
        
        # Keep more edges for dense graph
        kept_edges = sorted_edges[:min(len(sorted_edges), self.max_edges)]
        
        final_edges = [(src, dst) for _, (_, _, src, dst) in kept_edges]
        final_weights = [weight for _, (weight, _, _, _) in kept_edges]
        
        return final_edges, final_weights
    
    def _print_dense_graph_statistics(self, edges: List[Tuple[int, int]]):
        """Print dense graph statistics"""
        print(f"\n[DENSE Graph Statistics]")
        print(f"  - Total unique edges: {len(edges):,}")
        avg_degree = 2 * len(edges) / len(self.node_ids) if len(self.node_ids) > 0 else 0
        print(f"  - Average degree: {avg_degree:.2f}")
        density = len(edges) / (len(self.node_ids) * (len(self.node_ids) - 1) / 2) * 100
        print(f"  - Density: {density:.4f}%")
        print(f"  - Graph density level: {'VERY DENSE' if density > 5 else 'DENSE' if density > 1 else 'MODERATE'}")
    
    def to_pyg(self) -> Data:
        """Convert to PyTorch Geometric format with dense graph"""
        print("[Info] Converting DENSE graph to PyTorch Geometric format...")
        
        edges, edge_weights = self.build_dense_graph()
        
        # Create edge tensors
        if edges:
            edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
            
            # Make undirected and add self-loops
            edge_index, edge_attr = to_undirected(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=len(self.node_ids))
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.float32)
        
        # Create feature tensor
        x = self._create_feature_tensor()
        y = torch.tensor(self.y, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        print(f"\n[DENSE Graph Summary]")
        print(f"Nodes: {data.num_nodes:,}")
        print(f"Edges: {data.num_edges:,}")
        print(f"Features: {data.num_node_features}")
        print(f"Avg Degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Target distribution: {torch.bincount(data.y)}")
        print(f"Memory usage: Dense graph optimized for 16GB systems")
        
        self.memory_monitor.log_memory("Dense PyG Data Creation Complete")
        return data
    
    def _create_feature_tensor(self) -> torch.Tensor:
        """Create feature tensor with robust error handling"""
        print("[Info] Creating dense feature tensor...")
        
        try:
            feature_data = self.X_engineered.values.astype(np.float32)
            
            if not np.isfinite(feature_data).all():
                print("[Warning] Non-finite values detected, replacing with 0")
                feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            return torch.tensor(feature_data, dtype=torch.float32)
            
        except Exception as e:
            print(f"[Error] Feature tensor creation failed: {e}")
            return self._create_fallback_tensor()
    
    def _create_fallback_tensor(self) -> torch.Tensor:
        """Create fallback tensor column by column"""
        feature_list = []
        for col in self.X_engineered.columns:
            try:
                col_data = self.X_engineered[col].values.astype(np.float32)
                feature_list.append(col_data)
            except Exception:
                print(f"[Warning] Using zeros for column {col}")
                feature_list.append(np.zeros(len(self.X_engineered), dtype=np.float32))
        
        if not feature_list:
            raise RuntimeError("Could not create feature tensor")
        
        feature_data = np.column_stack(feature_list)
        return torch.tensor(feature_data, dtype=torch.float32)

############################################################################
# Model Evaluation Utilities
############################################################################

"""
Computes comprehensive metrics, generates reports, and visualizes results to assess the 
    model’s effectiveness in capturing organizational and individual factors influencing turnover.

Key Features:
- Enhanced class weighting strategies to better handle class imbalance.
- Comprehensive metric computation including AUC, PR AUC, F1, precision, recall, and confusion matrix.
- Optimal threshold determination for various performance metrics.
- Detailed classification report generation.

Flow:
1. Compute class weights using enhanced strategies.
2. Evaluate model to obtain predictions and probabilities.
3. Compute metrics based on predictions and true labels.
4. Determine optimal thresholds for F1, balanced accuracy, and recall.
5. Generate detailed classification report.

Which metrics to focus on:
- AUC and PR AUC for overall discrimination ability.
- F1 score for balance between precision and recall.
- Confusion matrix for detailed error analysis.

Evaluation Metrics:
    In the context of employee turnover prediction, these metrics help understand 
    how well the model identifies employees at risk of leaving, balancing false positives and false negatives effectively.
    By focusing on these key metrics, organizations can make informed decisions to improve employee retention strategies.
    Precision is crucial to avoid unnecessary interventions, while recall ensures that most at-risk employees are identified.
    Accurate predictions can lead to targeted retention efforts, ultimately reducing turnover rates and associated costs.
    Threshold tuning allows for customization based on organizational priorities, whether minimizing false positives or maximizing true positives.

"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    precision_recall_fscore_support, confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


def compute_enhanced_class_weights(y: np.ndarray, strategy: str = 'balanced_focal') -> torch.Tensor:
    """Enhanced class weighting strategies"""
    classes, counts = np.unique(y, return_counts=True)
    freq = {c: cnt for c, cnt in zip(classes, counts)}
    n0, n1 = freq.get(0, 0), freq.get(1, 0)
    
    if n0 == 0 or n1 == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float)
    
    if strategy == 'balanced_focal':
        # Focal loss style weighting with extra minority boost
        total = n0 + n1
        w0 = total / (2.0 * n0)
        w1 = total / (2.0 * n1) * 3.0  # Strong boost for minority class
    elif strategy == 'sqrt_balanced':
        # Square root balanced
        w0 = np.sqrt(n1 / n0)
        w1 = np.sqrt(n0 / n1) * 2.0
    else:  # standard balanced
        total = n0 + n1
        w0 = total / (2.0 * n0)
        w1 = total / (2.0 * n1)
    
    return torch.tensor([w0, w1], dtype=torch.float)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.threshold_metrics = {}
    
    def evaluate_model(self, model: torch.nn.Module, data, mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model and return predictions, probabilities, and labels"""
        model.eval()
        
        with torch.no_grad():
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                logits = model(data.x, data.edge_index, edge_weight=data.edge_attr)
            else:
                logits = model(data.x, data.edge_index)
                
            probs = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
            preds = logits[mask].argmax(dim=1).cpu().numpy()
            labels = data.y[mask].cpu().numpy()
            
            return probs, preds, labels
    
    def compute_metrics(self, labels: np.ndarray, probs: np.ndarray, 
                       threshold: float = 0.5) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        preds = (probs >= threshold).astype(int)
        
        try:
            auc = roc_auc_score(labels, probs)
            pr_auc = average_precision_score(labels, probs)
        except:
            auc = 0.5
            pr_auc = 0.277  # Default for imbalanced dataset
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='binary', zero_division=0
            )
        except:
            precision = recall = f1 = 0.0
        
        try:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            balanced_accuracy = (sensitivity + specificity) / 2
        except:
            specificity = sensitivity = balanced_accuracy = 0.0
        
        return {
            'auc': auc,
            'pr_auc': pr_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy
        }
    
    def find_optimal_thresholds(self, labels: np.ndarray, probs: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Find optimal thresholds for different metrics"""
        thresholds = np.linspace(0.05, 0.95, 50)
        
        best_f1 = (0, 0)
        best_balanced_acc = (0, 0)
        best_recall = (0, 0)
        
        for threshold in thresholds:
            metrics = self.compute_metrics(labels, probs, threshold)
            
            if metrics['f1'] > best_f1[1]:
                best_f1 = (threshold, metrics['f1'])
            
            if metrics['balanced_accuracy'] > best_balanced_acc[1]:
                best_balanced_acc = (threshold, metrics['balanced_accuracy'])
            
            if metrics['recall'] > best_recall[1]:
                best_recall = (threshold, metrics['recall'])
        
        return {
            'best_f1': best_f1,
            'best_balanced_acc': best_balanced_acc,
            'best_recall': best_recall
        }
    
    def generate_classification_report(self, labels: np.ndarray, probs: np.ndarray, 
                                     threshold: float = 0.5) -> str:
        """Generate detailed classification report"""
        preds = (probs >= threshold).astype(int)
        
        report = classification_report(
            labels, preds,
            target_names=["No Attrition", "Attrition"],
            digits=4
        )
        
        return report
    
    def plot_evaluation_curves(self, labels: np.ndarray, probs: np.ndarray, 
                             save_path: Optional[str] = None):
        """Plot ROC and PR curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        
        ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = average_precision_score(labels, probs)
        
        ax2.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
        ax2.axhline(y=labels.mean(), color='k', linestyle='--', alpha=0.5, 
                   label=f'Baseline ({labels.mean():.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Info] Evaluation curves saved to: {save_path}")
        
        plt.show()
    
    def comprehensive_evaluation(self, model: torch.nn.Module, data, test_mask: torch.Tensor,
                                report_dir: str, model_name: str) -> Dict[str, float]:
        """Run comprehensive evaluation and save results"""
        # Get predictions
        test_probs, test_preds, test_labels = self.evaluate_model(model, data, test_mask)
        
        # Compute metrics
        metrics = self.compute_metrics(test_labels, test_probs)
        
        # Find optimal thresholds
        threshold_results = self.find_optimal_thresholds(test_labels, test_probs)
        
        # Generate classification report
        class_report = self.generate_classification_report(test_labels, test_probs)
        
        # Create report directory
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        # Plot and save evaluation curves
        self.plot_evaluation_curves(
            test_labels, test_probs,
            save_path=Path(report_dir) / f"{model_name}_evaluation_curves.png"
        )
        
        # Print results
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*70)
        print(f"Final Results:")
        print(f"  Test ROC AUC: {metrics['auc']:.4f}")
        print(f"  Test PR AUC: {metrics['pr_auc']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        
        # Performance assessment
        if metrics['auc'] >= 0.65:
            print(f"  EXCELLENT! Matches XGBoost performance!")
        elif metrics['auc'] >= 0.60:
            print(f"  VERY GOOD! Close to target performance!")
        elif metrics['auc'] >= 0.55:
            print(f"  GOOD! Significant improvement over baseline!")
        else:
            print(f"  Needs further optimization")

        print(f"\nClassification Report:")
        print(class_report)
        
        print(f"\nOptimal Thresholds:")
        for metric, (threshold, score) in threshold_results.items():
            print(f"  {metric}: {score:.4f} @ threshold {threshold:.3f}")
        
        return metrics

"""
Dense GCN Pipeline
==================

Specialized pipeline for training GCN with dense organizational graphs.
Optimized for 16GB memory systems with enhanced organizational relationships.

Usage:
    python src/modelling/gnn/dense_gcn_pipeline.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

# Add src to path for local imports
script_path = Path(__file__).resolve()
src_path = script_path.parent.parent.parent
sys.path.append(str(src_path))


class DenseGCNTrainer:
    """Specialized trainer for dense GCN models"""
    
    def __init__(self, config):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.device = torch.device(config.device)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
        # Data splits
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        
        # Best model tracking
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        print(f"[Info] Dense GCN Trainer initialized with device: {self.device}")
        print(f"[Info] Max edges: {config.max_edges:,}")
        print(f"[Info] Similarity threshold: {config.similarity_threshold}")
    
    def setup_model(self, model: nn.Module, data):
        """Setup model and training components for dense graphs"""
        self.model = model.to(self.device)
        
        # Create data splits
        self.train_mask, self.val_mask, self.test_mask = self._create_data_splits(data)
        
        # Setup optimizer with gradient clipping
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Setup scheduler for dense graphs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.7,
            patience=20
        )
        
        # Setup loss function with enhanced class weights
        train_labels = data.y[self.train_mask].cpu().numpy()
        class_weights = compute_enhanced_class_weights(
            train_labels, 
            strategy=self.config.class_weight_strategy
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"[Info] Dense GCN model setup complete:")
        print(f"  - Parameters: {self.model.count_parameters():,}")
        print(f"  - Class weights: {class_weights}")
        print(f"  - Graph density: {data.num_edges / (data.num_nodes * (data.num_nodes - 1) / 2) * 100:.4f}%")
        
        return self.model
    
    def _create_data_splits(self, data):
        """Create stratified train/val/test splits"""
        idx = np.arange(data.num_nodes)
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            idx, 
            test_size=self.config.test_size,
            stratify=data.y.cpu().numpy(),
            random_state=self.config.seed
        )
        
        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=self.config.val_size,
            stratify=data.y[train_val_idx].cpu().numpy(),
            random_state=self.config.seed
        )
        
        # Create masks
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        print(f"[Info] Data splits - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        
        return train_mask.to(self.device), val_mask.to(self.device), test_mask.to(self.device)
    
    def train(self, data):
        """Train the dense GCN model"""
        data = data.to(self.device)
        
        print(f"[Info] Training dense GCN for {self.config.epochs} epochs...")
        print(f"[Info] Graph stats - Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")
        
        training_history = {"epochs": [], "train_loss": [], "val_auc": [], "lr": []}
        
        for epoch in range(1, self.config.epochs + 1):
            # Training step
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass with edge weights for GCN
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                out = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
            else:
                out = self.model(data.x, data.edge_index)
            
            loss = self.criterion(out[self.train_mask], data.y[self.train_mask])
            
            loss.backward()
            
            # Gradient clipping for stability with dense graphs
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
            
            # Validation and logging
            if epoch % 10 == 0 or epoch == 1:
                val_auc = self._evaluate(data, self.val_mask)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | LR: {current_lr:.6f}")
                
                # Update learning rate scheduler
                self.scheduler.step(val_auc)
                
                # Track best model
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping check
                if self.patience_counter >= self.config.patience:
                    print(f"[Info] Early stopping at epoch {epoch} (patience: {self.config.patience})")
                    break
                
                # Store history
                training_history["epochs"].append(epoch)
                training_history["train_loss"].append(loss.item())
                training_history["val_auc"].append(val_auc)
                training_history["lr"].append(current_lr)
        
        print(f"[Info] Training complete. Best Val AUC: {self.best_val_auc:.4f} at epoch {self.best_epoch}")
        return training_history
    
    def _evaluate(self, data, mask):
        """Evaluate model on given mask"""
        self.model.eval()
        with torch.no_grad():
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                out = self.model(data.x, data.edge_index, edge_weight=data.edge_attr)
            else:
                out = self.model(data.x, data.edge_index)
            
            probs = F.softmax(out[mask], dim=1)[:, 1].cpu().numpy()
            labels = data.y[mask].cpu().numpy()
            
            try:
                auc = roc_auc_score(labels, probs)
            except:
                auc = 0.5
            
            return auc

############################################################################
# Dense GCN Pipeline
############################################################################

"""
The DenseGCNPipeline class is the orchestrator of the entire workflow using a dense GCN.
It integrates data loading, graph construction, model training, and evaluation into a cohesive pipeline, optimized for capturing complex organizational relationships and individual features that influence turnover.

Key Features:
- Dense Graph Construction: Builds a rich organizational graph with multiple relationship types, optimized for memory efficiency.
- Specialized Dense GCN Trainer: Implements training strategies tailored for dense graphs, including advanced optimizers and learning rate schedulers.
- Comprehensive Evaluation: Utilizes enhanced evaluation metrics and reporting to assess model performance effectively.
- Memory Monitoring: Tracks memory usage throughout the pipeline to ensure efficient resource management.
- Configurable Parameters: Allows customization of key parameters such as max edges, similarity thresholds, and training hyperparameters.
- Robust Error Handling: Ensures stability and reliability across all stages of the pipeline.
- Detailed Reporting: Generates comprehensive reports and visualizations to facilitate analysis and interpretation of results.

Flow:
1. Data Loading: Reads and validates the input dataset.
2. Dense Graph Construction: Builds a dense organizational graph using various relationship types and filters edges intelligently.
3. Model Training: Trains the dense GCN model using the specialized trainer.
4. Model Evaluation: Assesses the trained model's performance on validation and test sets.
5. Reporting: Generates detailed reports and visualizations to summarize findings and insights.

"""


class DenseGCNPipeline:
    """Complete pipeline for dense GCN training"""
    
    def __init__(self, config=None, data_path=ENRICHED_FILE_PATH):
        self.config = config if config else DENSE_GCN_CONFIG
        self.data_path = data_path
        self.memory_monitor = MemoryMonitor()
        
        # Ensure report directory exists
        Path(self.config.report_dir).mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("DENSE GCN PIPELINE - Optimized for Rich Organizational Graphs")
        print("="*80)
        print(f" Data Path: {self.data_path}")
        print(f" Report Dir: {self.config.report_dir}")
        print(f" Max Edges: {self.config.max_edges:,}")
        print(f" Similarity Threshold: {self.config.similarity_threshold}")
        print(f" Device: {self.config.device}")
        print("="*80)
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete dense GCN pipeline"""
        pipeline_start = datetime.now()
        
        try:
            # Step 1: Load data
            print("\\n Step 1: Loading data...")
            df = self._load_data()
            
            # Step 2: Build dense graph
            print("\\n Step 2: Building DENSE organizational graph...")
            graph_data = self._build_dense_graph(df)
            
            # Step 3: Train dense GCN
            print("\\n Step 3: Training Dense GCN...")
            results = self._train_dense_gcn(graph_data)
            
            # Step 4: Generate reports
            print("\\n Step 4: Generating reports...")
            self._generate_reports(results, graph_data)
            
            pipeline_end = datetime.now()
            duration = pipeline_end - pipeline_start
            
            summary = {
                "pipeline_duration": str(duration),
                "final_auc": results.get("final_auc", 0),
                "best_val_auc": results.get("best_val_auc", 0),
                "graph_density": results.get("graph_density", 0),
                "peak_memory_mb": self.memory_monitor.get_peak_memory(),
                "config": self.config.to_dict()
            }
            
            self._print_final_results(summary)
            return summary
            
        except Exception as e:
            print(f"\\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def _load_data(self) -> pd.DataFrame:
        """Load and validate data"""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f" Loaded data shape: {df.shape}")
        print(f" Target distribution: {df['Attrition'].value_counts().to_dict()}")
        
        return df
    
    def _build_dense_graph(self, df: pd.DataFrame):
        """Build dense organizational graph"""
        self.memory_monitor.log_memory("Dense Graph Building Start")
        
        graph_builder = DenseGCNGraphBuilder(
            df_raw=df,
            max_edges=self.config.max_edges,
            similarity_threshold=self.config.similarity_threshold,
            top_features_count=self.config.top_features_count
        )
        
        graph_data = graph_builder.to_pyg()
        
        print(f" Dense graph built successfully:")
        print(f"   - Nodes: {graph_data.num_nodes:,}")
        print(f"   - Edges: {graph_data.num_edges:,}")
        print(f"   - Features: {graph_data.num_node_features}")
        print(f"   - Avg Degree: {graph_data.num_edges / graph_data.num_nodes:.2f}")
        print(f"   - Density: {graph_data.num_edges / (graph_data.num_nodes * (graph_data.num_nodes - 1) / 2) * 100:.4f}%")
        
        self.memory_monitor.log_memory("Dense Graph Building Complete")
        return graph_data
    
    def _train_dense_gcn(self, graph_data):
        """Train the dense GCN model"""
        # Create model
        model = create_model(self.config, graph_data.num_node_features)
        
        # Create trainer
        trainer = DenseGCNTrainer(self.config)
        
        # Setup and train
        trainer.setup_model(model, graph_data)
        training_history = trainer.train(graph_data)
        
        # Final evaluation
        evaluator = ModelEvaluator()
        if trainer.test_mask is not None:
            test_probs, test_preds, test_labels = evaluator.evaluate_model(
                model, graph_data, trainer.test_mask
            )
            final_metrics = evaluator.compute_metrics(test_labels, test_probs)
        else:
            final_metrics = {"auc": 0.0, "pr_auc": 0.0, "f1": 0.0}
        
        return {
            "final_auc": final_metrics["auc"],
            "final_pr_auc": final_metrics["pr_auc"],
            "final_f1": final_metrics["f1"],
            "best_val_auc": trainer.best_val_auc,
            "best_epoch": trainer.best_epoch,
            "training_history": training_history,
            "model_params": model.count_parameters(),
            "graph_density": graph_data.num_edges / (graph_data.num_nodes * (graph_data.num_nodes - 1) / 2) * 100
        }
    
    def _generate_reports(self, results: Dict[str, Any], graph_data):
        """Generate comprehensive reports"""
        # Save metrics
        metrics_file = Path(self.config.report_dir) / "dense_gcn_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save training history
        if "training_history" in results:
            history_file = Path(self.config.report_dir) / "dense_gcn_training_history.csv"
            pd.DataFrame(results["training_history"]).to_csv(history_file, index=False)
        
        # Save graph statistics
        graph_stats = {
            "nodes": int(graph_data.num_nodes),
            "edges": int(graph_data.num_edges),
            "features": int(graph_data.num_node_features),
            "avg_degree": float(graph_data.num_edges / graph_data.num_nodes),
            "density_percent": float(results["graph_density"]),
            "target_distribution": torch.bincount(graph_data.y).tolist()
        }
        
        stats_file = Path(self.config.report_dir) / "dense_graph_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(graph_stats, f, indent=2)
    
    def _print_final_results(self, summary: Dict[str, Any]):
        """Print final results"""
        print("\\n" + "="*80)
        print("DENSE GCN PIPELINE COMPLETE")
        print("="*80)
        
        auc = summary.get("final_auc", 0)
        val_auc = summary.get("best_val_auc", 0)
        density = summary.get("graph_density", 0)
        
        print(f" Duration: {summary.get('pipeline_duration', 'Unknown')}")
        print(f" Final Test AUC: {auc:.4f}")
        print(f" Best Val AUC: {val_auc:.4f}")
        print(f" Graph Density: {density:.4f}%")
        print(f" Peak Memory: {summary.get('peak_memory_mb', 0):.1f} MB")

        # Performance assessment
        benchmarks = {"XGBoost": 0.650, "CatBoost": 0.638, "Ensemble": 0.655}
        
        print("\\n Performance vs Benchmarks:")
        for model, benchmark in benchmarks.items():
            diff = auc - benchmark
            status = " BETTER" if diff > 0 else "📈 APPROACHING" if diff > -0.05 else "🔧 NEEDS WORK"
            print(f"  vs {model}: {diff:+.4f} ({status})")
        
        if auc >= 0.63:
            print("\\n SUCCESS! Dense GCN achieved competitive performance with traditional ML!")
        elif auc >= 0.58:
            print("\\n GOOD PROGRESS! Dense GCN shows promise, keep optimizing!")
        else:
            print("\\n More optimization needed. Consider:")
            print("   - Increasing graph density further")
            print("   - Adding more organizational relationship types")
            print("   - Tuning hyperparameters")
        
        print(f"\\n Reports saved to: {self.config.report_dir}")
        print("="*80)


def main():
    """Main entry point"""
    pipeline = DenseGCNPipeline()
    results = pipeline.run_pipeline()
    return results


if __name__ == "__main__":
    main()


# How to run:
# python src/modelling/gnn_model_enhanced_v0.py
