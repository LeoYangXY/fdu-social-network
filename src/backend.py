from flask import Flask, render_template, request, jsonify, send_from_directory
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # ✅ 关键修复：禁用 GUI 后端
import matplotlib.pyplot as plt
import community as community_louvain
import pandas as pd
from collections import Counter
import numpy as np
import base64
from io import BytesIO
import json
import os
import hashlib
import time
from datetime import datetime, timedelta
import threading
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
import gc
import random

# 设置项目根目录为当前文件所在目录的父目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'templates')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')

app = Flask(__name__, 
           template_folder=TEMPLATE_DIR,
           static_folder=STATIC_DIR)
app.config['SECRET_KEY'] = 'social_network_analyzer_key'

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据集配置 - 目前只使用facebook数据集
DATASETS = {
    "facebook": os.path.join(PROJECT_ROOT, "data", "facebook_combined.txt"),
}

# 缓存配置
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 分析进度存储
analysis_progress = {}
executor = ThreadPoolExecutor(max_workers=2)  # 限制并发线程数

class ProgressTracker:
    def __init__(self, total_steps, task_id):
        self.total_steps = total_steps
        self.current_step = 0
        self.task_id = task_id
        self.start_time = time.time()
        self.progress = 0
        self.status = "初始化"
        self.details = ""
        self.completed = False
        self.result = None
        analysis_progress[task_id] = self
    
    def update(self, step, status, details=""):
        self.current_step = step
        self.status = status
        self.details = details
        self.progress = int((step / self.total_steps) * 100)
        analysis_progress[self.task_id] = self
        logger.info(f"Task {self.task_id}: {status} - {self.progress}%")
    
    def finish(self, result=None):
        self.progress = 100
        self.status = "完成"
        self.result = result
        self.completed = True
        analysis_progress[self.task_id] = self

# ✅【关键修改】：用 JSON 替代 pickle 的 CacheManager
class CacheManager:
    def __init__(self, cache_dir=CACHE_DIR, ttl_hours=24):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self.lock = threading.Lock()
    
    def _get_cache_key(self, dataset_path, params):
        """生成缓存键"""
        version = "v1_json"
        cache_input = f"{dataset_path}_{str(params)}_{version}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key):
        return os.path.join(self.cache_dir, f"{cache_key}.json")  # .json 而非 .pkl
    
    def get(self, dataset_path, params):
        """获取缓存结果"""
        cache_key = self._get_cache_key(dataset_path, params)
        cache_file = self._get_cache_file_path(cache_key)
        
        if not os.path.exists(cache_file):
            logger.info(f"Cache miss for {dataset_path}")
            return None
        
        # 检查是否过期
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime > self.ttl_seconds:
            try:
                os.remove(cache_file)
            except:
                pass
            logger.info(f"Cache expired for {dataset_path}")
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            logger.info(f"Cache hit for {dataset_path}")
            return result
        except (json.JSONDecodeError, OSError, ValueError, KeyError) as e:
            logger.warning(f"Cache file corrupted: {e}. Removing it.")
            try:
                os.remove(cache_file)
            except:
                pass
            return None
    
    def put(self, dataset_path, params, result):
        """保存结果到缓存"""
        cache_key = self._get_cache_key(dataset_path, params)
        cache_file = self._get_cache_file_path(cache_key)
        
        safe_result = {
            'global_metrics': result.get('global_metrics', {}),
            'opinion_leaders': result.get('opinion_leaders', []),
            'community_info': result.get('community_info', {}),
            'visualizations': result.get('visualizations', {}),
            'analysis_time': result.get('analysis_time', datetime.now().isoformat()),
            'cached': False,
            'task_id': result.get('task_id', cache_key)
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(safe_result, f, ensure_ascii=False)
            logger.info(f"Cached result for {dataset_path}")
        except Exception as e:
            logger.error(f"Cache save failed: {e}")

def run_analysis_task(task_id, dataset_path, top_k):
    """在后台线程中执行分析任务"""
    try:
        analyzer = SocialNetworkAnalyzer(dataset_path)
        
        if not analyzer.load_graph(analysis_progress.get(task_id)):
            return {'error': 'Failed to load graph data'}
        
        global_metrics = analyzer.compute_global_metrics(analysis_progress.get(task_id))
        opinion_leaders = analyzer.identify_opinion_leaders(top_k=top_k, progress_tracker=analysis_progress.get(task_id))
        community_info = analyzer.detect_communities(progress_tracker=analysis_progress.get(task_id))
        additional_metrics = analyzer.compute_additional_metrics(progress_tracker=analysis_progress.get(task_id))
        visualizations = analyzer.generate_visualizations(progress_tracker=analysis_progress.get(task_id))
        
        all_metrics = {**global_metrics, **additional_metrics}
        
        result = {
            'global_metrics': all_metrics,
            'opinion_leaders': opinion_leaders.to_dict(orient='records') if opinion_leaders is not None else [],
            'community_info': community_info,
            'visualizations': visualizations,
            'analysis_time': datetime.now().isoformat(),
            'cached': False,
            'task_id': task_id
        }
        
        cache_manager.put(dataset_path, {'top_k': top_k}, result)
        
        if task_id in analysis_progress:
            analysis_progress[task_id].finish(result)
        
        logger.info(f"Analysis completed for task {task_id}")
        return result
    except Exception as e:
        logger.error(f"Analysis error for task {task_id}: {str(e)}")
        if task_id in analysis_progress:
            analysis_progress[task_id].finish({'error': str(e)})
        return {'error': str(e)}

class SocialNetworkAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.G = None
        self.partition = None
        self.leaders_df = None
        self.community_stats = None
        
    def load_graph(self, progress_tracker=None):
        if progress_tracker:
            progress_tracker.update(1, "加载图数据", f"从 {self.dataset_path} 加载")
        
        logger.info(f"Loading graph from {self.dataset_path}...")
        try:
            if any(x in self.dataset_path.lower() for x in ['facebook', 'amazon', 'youtube']):
                self.G = nx.read_edgelist(self.dataset_path, nodetype=int)
            elif 'vote' in self.dataset_path.lower():
                self.G = nx.read_edgelist(self.dataset_path, nodetype=int)
            else:
                self.G = nx.read_edgelist(self.dataset_path, nodetype=str)
            
            if self.G.is_directed():
                self.G = self.G.to_undirected()
            
            logger.info(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
            return True
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
            return False
    
    def identify_opinion_leaders(self, top_k=10, progress_tracker=None):
        if progress_tracker:
            progress_tracker.update(2, "计算度中心性", "计算节点的度中心性")
        
        try:
            degree_cent = nx.degree_centrality(self.G)
            
            if progress_tracker:
                progress_tracker.update(3, "计算介数中心性", "计算节点的介数中心性")
            
            # 对大图使用采样
            sample_size = min(1000, self.G.number_of_nodes())
            betweenness_cent = nx.betweenness_centrality(self.G, k=sample_size, seed=42)
            
            if progress_tracker:
                progress_tracker.update(4, "计算接近中心性", "计算节点的接近中心性")
            
            closeness_cent = nx.closeness_centrality(self.G)
            
            if progress_tracker:
                progress_tracker.update(5, "计算特征向量中心性", "计算节点的特征向量中心性")
            
            # 对大图限制迭代次数
            if self.G.number_of_nodes() > 5000:
                eigenvector_cent = nx.eigenvector_centrality(self.G, max_iter=100, tol=1e-3)
            else:
                eigenvector_cent = nx.eigenvector_centrality(self.G, max_iter=200, tol=1e-4)
            
            centrality_dict = {
                'degree': degree_cent,
                'betweenness': betweenness_cent,
                'closeness': closeness_cent,
                'eigenvector': eigenvector_cent
            }
            
            leaders = {}
            scores = {}
            
            for name, cent in centrality_dict.items():
                top_nodes = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:top_k]
                leaders[name] = [node for node, score in top_nodes]
                scores[name] = [score for node, score in top_nodes]
            
            self.leaders_df = pd.DataFrame({
                'node': leaders['degree'],
                'degree_score': scores['degree'],
                'betweenness_score': scores['betweenness'],
                'closeness_score': scores['closeness'],
                'eigenvector_score': scores['eigenvector']
            })
            
            if progress_tracker:
                progress_tracker.update(6, "意见领袖识别完成", f"已识别前{top_k}名意见领袖")
            
            return self.leaders_df
        except Exception as e:
            logger.error(f"Error computing opinion leaders: {str(e)}")
            return None
    
    def compute_global_metrics(self, progress_tracker=None):
        if progress_tracker:
            progress_tracker.update(7, "计算全局网络指标", "分析网络密度、聚类系数等")
        
        try:
            metrics = {
                'nodes': self.G.number_of_nodes(),
                'edges': self.G.number_of_edges(),
                'density': round(nx.density(self.G), 6),
                'average_clustering': round(nx.average_clustering(self.G), 6),
                'average_degree': round(2 * self.G.number_of_edges() / self.G.number_of_nodes(), 2),
                'transitivity': round(nx.transitivity(self.G), 6),
                'assortativity': round(nx.degree_assortativity_coefficient(self.G), 6)
            }
            
            if progress_tracker:
                progress_tracker.update(8, "分析连通性", "检查网络连通分量")
            
            components = list(nx.connected_components(self.G))
            metrics['connected_components'] = len(components)
            metrics['largest_component_size'] = max(len(c) for c in components) if components else 0
            
            if progress_tracker:
                progress_tracker.update(9, "计算路径长度", "计算直径、半径等路径指标")
            
            if nx.is_connected(self.G):
                try:
                    if self.G.number_of_nodes() > 5000:
                        # 对大图使用近似算法
                        metrics['diameter'] = "N/A (approx too expensive)"
                        metrics['radius'] = "N/A (approx too expensive)"
                        metrics['avg_shortest_path'] = "N/A (approx too expensive)"
                    else:
                        metrics['diameter'] = nx.diameter(self.G)
                        metrics['radius'] = nx.radius(self.G)
                        metrics['avg_shortest_path'] = round(nx.average_shortest_path_length(self.G), 4)
                except nx.NetworkXNoPath:
                    metrics['diameter'] = "N/A (disconnected components)"
                    metrics['radius'] = "N/A (disconnected components)"
                    metrics['avg_shortest_path'] = "N/A (disconnected components)"
                except Exception as e:
                    logger.warning(f"Could not compute path metrics: {e}")
                    metrics['diameter'] = "N/A (computation too expensive)"
                    metrics['radius'] = "N/A (computation too expensive)"
                    metrics['avg_shortest_path'] = "N/A (computation too expensive)"
            else:
                metrics['diameter'] = "N/A (disconnected)"
                metrics['radius'] = "N/A (disconnected)"
                metrics['avg_shortest_path'] = "N/A (disconnected)"
            
            return metrics
        except Exception as e:
            logger.error(f"Error computing global metrics: {str(e)}")
            return {}
    
    def detect_communities(self, progress_tracker=None):
        if progress_tracker:
            progress_tracker.update(10, "运行社区检测", "使用Louvain算法检测社区结构")
        
        logger.info("Running community detection...")
        try:
            self.partition = community_louvain.best_partition(self.G, random_state=42, resolution=1.0)
            modularity = community_louvain.modularity(self.partition, self.G)
            
            community_stats = Counter(self.partition.values())
            num_communities = len(community_stats)
            
            community_details = []
            for comm_id in range(num_communities):
                nodes_in_comm = [node for node, comm in self.partition.items() if comm == comm_id]
                subgraph = self.G.subgraph(nodes_in_comm)
                details = {
                    'id': comm_id,
                    'size': len(nodes_in_comm),
                    'internal_edges': subgraph.number_of_edges(),
                    'density': round(nx.density(subgraph), 6) if len(nodes_in_comm) > 1 else 0
                }
                community_details.append(details)
            
            self.community_stats = {
                'partition': self.partition,
                'modularity': round(modularity, 6),
                'num_communities': num_communities,
                'community_sizes': dict(community_stats),
                'details': community_details
            }
            
            if progress_tracker:
                progress_tracker.update(11, "社区检测完成", f"发现{num_communities}个社区，模块度:{modularity:.4f}")
            
            return self.community_stats
        except Exception as e:
            logger.error(f"Error detecting communities: {str(e)}")
            return {}
    
    def compute_additional_metrics(self, progress_tracker=None):
        if progress_tracker:
            progress_tracker.update(12, "计算额外指标", "计算度分布、H指数等")
        
        try:
            additional_metrics = {}
            
            degrees = [d for n, d in self.G.degree()]
            additional_metrics['max_degree'] = max(degrees) if degrees else 0
            additional_metrics['min_degree'] = min(degrees) if degrees else 0
            additional_metrics['avg_degree'] = sum(degrees) / len(degrees) if degrees else 0
            
            degree_counts = Counter(degrees)
            additional_metrics['degree_distribution'] = dict(degree_counts)
            
            h_index = 0
            sorted_degrees = sorted(degrees, reverse=True)
            for i, deg in enumerate(sorted_degrees):
                if deg >= i + 1:
                    h_index = i + 1
                else:
                    break
            additional_metrics['h_index'] = h_index
            
            return additional_metrics
        except Exception as e:
            logger.error(f"Error computing additional metrics: {str(e)}")
            return {}
        
    def generate_visualizations(self, progress_tracker=None):
        try:
            visualizations = {}
            n_nodes = self.G.number_of_nodes()
            
            # === 1. 度分布图（始终生成，很快）===
            if progress_tracker:
                progress_tracker.update(13, "生成度分布图", "快速绘制度分布")

            fig, ax = plt.subplots(figsize=(6, 4))
            degrees = [d for n, d in self.G.degree()]
            max_bins = min(30, len(set(degrees)))
            ax.hist(degrees, bins=max_bins, color='skyblue', alpha=0.8)
            ax.set_xlabel("Degree")
            ax.set_ylabel("Count")
            ax.set_title("Degree Distribution")
            visualizations['degree_dist'] = self.fig_to_base64(fig)
            plt.close(fig)

            # === 2. 社区规模分布（如果检测了社区）===
            if hasattr(self, 'community_stats') and self.community_stats:
                if progress_tracker:
                    progress_tracker.update(14, "生成社区规模分布", "快速绘制社区大小")
                sizes = list(self.community_stats['community_sizes'].values())
                fig, ax = plt.subplots(figsize=(6, 4))
                max_bins = min(15, len(sizes))
                ax.hist(sizes, bins=max_bins, color='salmon', alpha=0.8)
                ax.set_xlabel("Community Size")
                ax.set_ylabel("Count")
                ax.set_title("Community Size Dist")
                visualizations['community_size_dist'] = self.fig_to_base64(fig)
                plt.close(fig)

            # === 3. 中心性对比图（如果有意见领袖）===
            if self.leaders_df is not None:
                if progress_tracker:
                    progress_tracker.update(15, "生成中心性对比", "快速绘制Top节点指标")
                top_n = min(8, len(self.leaders_df))
                fig, ax = plt.subplots(figsize=(7, 4))
                x = range(top_n)
                for col in ['degree_score', 'betweenness_score']:
                    if col in self.leaders_df.columns:
                        ax.plot(x, self.leaders_df[col].values[:top_n], label=col.split('_')[0], marker='o')
                ax.set_title("Top Nodes: Degree vs Betweenness")
                ax.legend()
                visualizations['centrality_comparison'] = self.fig_to_base64(fig)
                plt.close(fig)

            # === 4. 意见领袖与社区结构可视化（完整图，社区按颜色区分，优化布局）===
            if progress_tracker:
                progress_tracker.update(16, "生成意见领袖社区图", f"处理{n_nodes}个节点")

            # 使用原始图（不采样）
            subG = self.G
            sub_partition = self.partition if hasattr(self, 'partition') and self.partition else {}

            # 使用kamada_kawai_layout布局，使社区结构更明显
            if n_nodes <= 5000:
                # 使用 Kamada-Kawai 布局，能更好保持社区结构
                # 移除seed参数以兼容低版本NetworkX
                pos = nx.kamada_kawai_layout(subG)
            else:
                # 对超大图使用 spring_layout + 更多迭代
                pos = nx.spring_layout(subG, k=0.8, iterations=50, seed=42)

            # 创建意见领袖与社区结构结合图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制社区节点（根据社区着色）
            if sub_partition:
                # 获取所有社区ID并排序
                comm_ids = sorted(list(set(sub_partition.values())))
                
                # 为每个社区分配一个颜色
                colors = plt.cm.tab20(np.linspace(0, 1, len(comm_ids)))
                
                # 创建颜色映射
                color_map = {comm_id: colors[i] for i, comm_id in enumerate(comm_ids)}
                node_colors = [color_map[sub_partition[n]] if n in sub_partition else 'lightgray' for n in subG.nodes()]
                
                # 绘制普通节点 - 按社区着色
                normal_nodes = [n for n in subG.nodes() if self.leaders_df is None or n not in self.leaders_df['node'].tolist()]
                if normal_nodes:
                    # 获取每个节点对应的社区ID
                    normal_node_colors = [color_map[sub_partition[n]] if n in sub_partition else 'lightgray' for n in normal_nodes]
                    nx.draw_networkx_nodes(
                        subG, pos,
                        nodelist=normal_nodes,
                        node_color=normal_node_colors,      # ✅ 使用社区颜色
                        node_size=10,
                        alpha=0.6
                    )
            else:
                # 如果没有社区信息，才用统一灰色
                nx.draw_networkx_nodes(subG, pos, node_size=10, alpha=0.6, node_color='lightgray')
            
            # 绘制意见领袖（覆盖在上面，标红标大）
            if self.leaders_df is not None:
                leader_nodes = [n for n in self.leaders_df['node'].tolist() if n in subG]
                if leader_nodes:
                    # 绘制意见领袖节点（红色，更大）
                    nx.draw_networkx_nodes(subG, pos, nodelist=leader_nodes,
                                         node_color='red',
                                         node_size=100, 
                                         edgecolors='white', 
                                         linewidths=2)
            
            # 绘制边（只绘制部分边以避免遮挡）
            edges = list(subG.edges())
            if len(edges) > 1000:
                edges = random.sample(edges, int(len(edges) * 0.1))
            
            nx.draw_networkx_edges(subG, pos, edgelist=edges, alpha=0.1, width=0.2)
            
            plt.axis('off')
            plt.title(f"Opinion Leaders (Red) + Communities - {len(subG.nodes())} nodes", fontsize=14, fontweight='bold')
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightgray', label='Regular Nodes'),
                Patch(facecolor='red', label='Opinion Leaders')
            ]
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            visualizations['opinion_leaders_communities'] = self.fig_to_base64(fig)
            plt.close(fig)

            # === 5. 网络结构图（仅对较小的图）===
            if n_nodes <= 1000:
                if progress_tracker:
                    progress_tracker.update(17, "生成网络结构图", f"节点数: {n_nodes}")

                pos = nx.spring_layout(self.G, k=1, iterations=20, seed=42)

                fig, ax = plt.subplots(figsize=(6, 6))
                nx.draw_networkx_nodes(self.G, pos, node_size=5, alpha=0.6, node_color='steelblue')
                nx.draw_networkx_edges(self.G, pos, alpha=0.1, width=0.2)
                plt.axis('off')
                plt.title("Network Structure", fontsize=10)
                visualizations['network_structure'] = self.fig_to_base64(fig)
                plt.close(fig)

            if progress_tracker:
                progress_tracker.update(18, "可视化完成", "绘图结束")

            return visualizations

        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return {}
    
    def fig_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return img_str

# 初始化缓存管理器
cache_manager = CacheManager(ttl_hours=24)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Template not found: {e}")
        return f'''
        <html>
        <head><title>Social Network Analysis Platform</title></head>
        <body>
            <h1>🌐 社交网络分析平台</h1>
            <p>后端API服务正在运行</p>
            <p>API端点:</p>
            <ul>
                <li><a href="/api/status">/api/status</a> - 系统状态</li>
                <li><a href="/api/datasets">/api/datasets</a> - 可用数据集</li>
            </ul>
            <p>请确保前端文件存在于 templates/index.html</p>
            <p>错误详情: {str(e)}</p>
        </body>
        </html>
        '''

@app.route('/api/datasets')
def get_datasets():
    return jsonify(list(DATASETS.keys()))

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        dataset_name = data.get('dataset', 'facebook')
        top_k = data.get('top_k', 10)
        use_cache = data.get('use_cache', True)
        
        if dataset_name not in DATASETS:
            return jsonify({'error': 'Invalid dataset'}), 400
        
        dataset_path = DATASETS[dataset_name]
        
        if use_cache:
            cached_result = cache_manager.get(dataset_path, {'top_k': top_k})
            if cached_result:
                # ✅ 即使缓存命中，也创建一个"已完成"的任务记录
                task_id = str(uuid.uuid4())
                progress_tracker = ProgressTracker(18, task_id)
                progress_tracker.finish(result=cached_result)
                
                logger.info(f"Returning cached result as completed task for {dataset_name}")
                return jsonify({
                    'task_id': task_id,
                    'status': 'completed',
                    'cached': True,
                    'message': 'Analysis retrieved from cache.'
                })
        
        # 缓存未命中：启动新任务
        task_id = str(uuid.uuid4())
        progress_tracker = ProgressTracker(18, task_id)
        executor.submit(run_analysis_task, task_id, dataset_path, top_k)
        
        return jsonify({
            'task_id': task_id,
            'status': 'started',
            'message': 'Analysis started. Use /api/progress/<task_id> to check progress.'
        })
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
@app.route('/api/progress/<task_id>')
def get_progress(task_id):
    if task_id in analysis_progress:
        progress_obj = analysis_progress[task_id]
        response_data = {
            'progress': progress_obj.progress,
            'status': progress_obj.status,
            'details': progress_obj.details,
            'current_step': progress_obj.current_step,
            'total_steps': progress_obj.total_steps,
            'completed': progress_obj.completed
        }
        
        if progress_obj.completed and progress_obj.result:
            response_data['result'] = progress_obj.result
        
        return jsonify(response_data)
    else:
        return jsonify({
            'error': 'Task not found',
            'progress': 0,
            'status': 'Unknown',
            'details': '',
            'completed': False
        }), 404

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    try:
        import shutil
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    cache_files = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    active_tasks = len(analysis_progress)
    return jsonify({
        'status': 'running',
        'cache_entries': cache_files,
        'datasets_available': list(DATASETS.keys()),
        'active_analysis_tasks': active_tasks,
        'server_time': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("后端API服务启动中... 请访问 http://localhost:5000 查看前端或 http://localhost:5000/api/status 查看API状态")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"模板目录: {TEMPLATE_DIR}")
    print(f"静态目录: {STATIC_DIR}")
    app.run(debug=False, host='0.0.0.0', port=5000)