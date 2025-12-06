import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import pandas as pd
from collections import Counter
import os
import shutil  

# Remove old results and create a fresh results directory
if os.path.exists("results"):
    shutil.rmtree("results")
os.makedirs("results", exist_ok=True)


def load_facebook_graph(filepath: str) -> nx.Graph:
    """Load Facebook graph from edge list."""
    print("✅ Loading Facebook graph...")
    G = nx.read_edgelist(filepath, nodetype=int)
    return nx.Graph(G)


def visualize_raw_graph(G: nx.Graph):
    """Visualize raw graph (nodes only)."""
    print("🎨 Visualizing raw graph (nodes only)...")
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color='lightblue', alpha=0.8)
    plt.title("Facebook Social Network (Raw Structure)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("results/1_raw_graph.png", dpi=150)
    plt.show()


def analyze_influential_nodes(G: nx.Graph, top_k: int = 10):
    """Identify opinion leaders using centrality measures."""
    print(f"\n🔍 Computing centrality measures for top-{top_k} opinion leaders...")

    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G, k=min(1000, G.number_of_nodes()), seed=42)
    closeness_cent = nx.closeness_centrality(G)
    eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)

    centrality_dict = {
        'degree': degree_cent,
        'betweenness': betweenness_cent,
        'closeness': closeness_cent,
        'eigenvector': eigenvector_cent
    }

    leaders = {}
    for name, cent in centrality_dict.items():
        top_nodes = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:top_k]
        leaders[name] = [node for node, score in top_nodes]
        print(f"  Top-{top_k} {name} centrality nodes: {leaders[name]}")

    # Save to CSV
    df_leaders = pd.DataFrame(leaders)
    df_leaders.to_csv("results/opinion_leaders_top10.csv", index=False)
    print(f"\n📊 Opinion leader table saved to: results/opinion_leaders_top10.csv")
    print(df_leaders.to_string(index=False))

    # Degree distribution
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Node Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution (Log Scale)")
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/2_degree_distribution.png", dpi=150)
    plt.show()

    return df_leaders


def compute_global_metrics(G: nx.Graph):
    """Compute global network metrics."""
    print("\n🌐 Computing global network metrics...")

    metrics = {}
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    metrics['average_clustering'] = nx.average_clustering(G)

    if nx.is_connected(G):
        metrics['diameter'] = nx.diameter(G)
        metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
    else:
        metrics['diameter'] = "N/A (disconnected)"
        metrics['avg_shortest_path'] = "N/A (disconnected)"

    print("📌 Global Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Save metrics to text file
    with open("results/global_metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print("📝 Global metrics saved to: results/global_metrics.txt")
    return metrics


def perform_community_detection(G: nx.Graph):
    """Perform community detection using Louvain algorithm."""
    print("\n🧩 Running Louvain community detection...")
    partition = community_louvain.best_partition(G, random_state=42)
    num_communities = max(partition.values()) + 1
    modularity = community_louvain.modularity(partition, G)

    print(f"  Number of communities: {num_communities}")
    print(f"  Modularity: {modularity:.4f}")

    # --- Visualization with updated cmap API ---
    print("🎨 Visualizing community structure...")
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
    cmap = plt.get_cmap('tab20', num_communities)  

    plt.figure(figsize=(14, 14))
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=partition.keys(),
        node_color=list(partition.values()),
        cmap=cmap,
        node_size=15,
        alpha=0.9
    )
    plt.title(f"Facebook Community Structure (Louvain, {num_communities} communities)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("results/3_communities.png", dpi=150)
    plt.show()

    # --- Community size distribution ---
    print("📊 Plotting community size distribution...")
    community_sizes = Counter(partition.values())
    sizes = list(community_sizes.values())
    plt.figure(figsize=(8, 6))
    plt.hist(sizes, bins=20, color='salmon', edgecolor='black', alpha=0.7)
    plt.xlabel("Community Size (Number of Nodes)")
    plt.ylabel("Number of Communities")
    plt.title("Community Size Distribution")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/4_community_size_distribution.png", dpi=150)
    plt.show()

    return partition, modularity


def visualize_communities_with_leaders(G: nx.Graph, partition: dict, leaders_df: pd.DataFrame, top_k_per_metric=5):
    """
    Visualize community structure and highlight opinion leaders in red.
    
    Parameters:
        G: Graph
        partition: {node: community_id}
        leaders_df: DataFrame with columns ['degree', 'betweenness', 'closeness', 'eigenvector']
        top_k_per_metric: 只取每种中心性 Top-K，避免红点过多（默认5）
    """
    print("\n🎨 Visualizing communities with opinion leaders highlighted in RED...")
    
    # Step 1: 获取所有唯一的 Top-K 意见领袖（取并集）
    leader_nodes = set()
    for col in leaders_df.columns:
        leader_nodes.update(leaders_df[col].head(top_k_per_metric).tolist())
    leader_nodes = list(leader_nodes)
    print(f"  Highlighting {len(leader_nodes)} unique opinion leaders (Top-{top_k_per_metric} from each metric)")

    # Step 2: 使用与社区图相同的布局（确保位置一致）
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
    
    # Step 3: 绘制普通节点（按社区着色）
    num_communities = max(partition.values()) + 1
    cmap = plt.get_cmap('tab20', num_communities)
    
    plt.figure(figsize=(14, 14))
    # 普通节点（非意见领袖）
    non_leader_nodes = [node for node in G.nodes() if node not in leader_nodes]
    non_leader_colors = [partition[node] for node in non_leader_nodes]
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=non_leader_nodes,
        node_color=non_leader_colors,
        cmap=cmap,
        node_size=10,
        alpha=0.7
    )
    
    # 意见领袖节点（红色，更大）
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=leader_nodes,
        node_color='red',
        node_size=80,  
        alpha=1.0,
        edgecolors='black', 
        linewidths=0.8
    )
    
    plt.title(f"Communities with Opinion Leaders Highlighted (Red, Top-{top_k_per_metric} per metric)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("results/5_communities_with_leaders.png", dpi=150)
    plt.show()

# ==================== Main Execution ====================
if __name__ == "__main__":
    FILE_PATH = "facebook_combined.txt"
    
    G = load_facebook_graph(FILE_PATH)
    
    #加载原图
    visualize_raw_org = visualize_raw_graph(G)
    
    #分析意见领袖
    leaders_df = analyze_influential_nodes(G, top_k=10)
    
    #计算全局指标
    global_metrics = compute_global_metrics(G)
    
    #划分社区（使用Louvain 社区发现算法）
    partition, mod = perform_community_detection(G)
    
    #意见领袖可视化
    visualize_communities_with_leaders(G, partition, leaders_df, top_k_per_metric=5)

    print("\n✅ All results saved in 'results/' folder!")