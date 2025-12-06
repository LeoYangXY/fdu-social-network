```markdown
# Social Network Analysis on Facebook Ego-Networks

This project performs **topological analysis** on the [Facebook social circles dataset](http://snap.stanford.edu/data/ego-Facebook.html) from SNAP. It includes:
- Opinion leader mining via centrality measures
- Community detection using Louvain algorithm
- Network structure visualization and metrics

All results are saved in the `results/` folder after running the script.

---

## 📦 Environment Setup

### Option 1: Using `conda` (Recommended)

1. Create and activate a new conda environment:
   ```bash
   conda create -n social_network python=3.9
   conda activate social_network
   ```

2. Install dependencies:
   ```bash
   conda install networkx matplotlib pandas
   pip install python-louvain
   ```

> 💡 Note: `python-louvain` (provides `community` module) is only available via `pip`.

### Option 2: Using `requirements.txt`

If you prefer pure pip:
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
networkx>=2.8
matplotlib>=3.5
pandas>=1.4
python-louvain>=0.15
```

---

## 🚀 How to Run

1. **Download the dataset**:
   - Go to [http://snap.stanford.edu/data/ego-Facebook.html](http://snap.stanford.edu/data/ego-Facebook.html)
   - Download `facebook_combined.txt.gz`
   - Extract it to get `facebook_combined.txt` in the project root

2. **Run the analysis**:
   ```bash
   python question_b.py
   ```

3. **View results**:
   - All outputs (images, CSV, metrics) are saved in the `results/` folder

---

## 📊 Code Workflow

The script executes the following steps **in order**:

1. **Load Graph**  
   - Reads `facebook_combined.txt` (4039 nodes, 88234 edges) as an undirected graph.

2. **Visualize Raw Structure**  
   - Plots all nodes (no edges) to show overall topology → `1_raw_graph.png`

3. **Opinion Leader Mining**  
   - Computes 4 centrality measures:
     - **Degree**: Direct connections
     - **Betweenness**: Control over information flow
     - **Closeness**: Proximity to all others
     - **Eigenvector**: Influence via high-status neighbors
   - Outputs Top-10 nodes per metric to console and `opinion_leaders_top10.csv`
   - Plots degree distribution → `2_degree_distribution.png`

4. **Global Network Metrics**  
   - Computes: node/edge count, density, clustering coefficient, diameter, average path length
   - Saves to `global_metrics.txt`

5. **Community Detection (Louvain)**  
   - Partitions network into communities to maximize modularity
   - Visualizes colored communities → `3_communities.png`
   - Plots community size distribution → `4_community_size_distribution.png`

6. **Highlight Opinion Leaders in Communities**  
   - Replots community structure with **Top-5 opinion leaders per metric** marked in **red**
   - Output: `5_communities_with_leaders.png`

---

## 📁 Output Files

After running, the `results/` folder contains:

| File | Description |
|------|-------------|
| `1_raw_graph.png` | Raw network (nodes only) |
| `2_degree_distribution.png` | Log-scale degree histogram |
| `3_communities.png` | Community structure (Louvain) |
| `4_community_size_distribution.png` | Histogram of community sizes |
| `5_communities_with_leaders.png` | Communities with opinion leaders in red |
| `opinion_leaders_top10.csv` | Top-10 nodes by each centrality |
| `global_metrics.txt` | Network-level statistics |

---

## 📚 Dataset Source

- **Name**: Facebook Ego-Networks
- **Nodes**: 4039
- **Edges**: 88234
- **Source**: [J. McAuley and J. Leskovec, NIPS 2012](https://arxiv.org/abs/1207.0192)
- **Link**: http://snap.stanford.edu/data/ego-Facebook.html

> ⚠️ Note: The network is a **static snapshot**; temporal evolution analysis is not possible.

---

## 📝 License

This project is for educational and research purposes only.
```