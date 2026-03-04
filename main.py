"""
LLM Strategy Discovery - Kaggle Optimized Version
Designed for H100 80GB GPU with disk space constraints

Features:
- Single file for easy Kaggle deployment
- 4-bit quantization for large models
- Qwen-32B support
- Memory-efficient processing
- Complete pipeline from data collection to analysis
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

# ML
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the experiment"""
    
    # Model settings
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"  # Default model
    USE_QUANTIZATION = True  # Use 4-bit quantization to save memory
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Game settings
    OPPONENT_STRATEGIES = ['TFT', 'AllC', 'AllD', 'WSLS', 'Random', 'Grudger']
    GAMES_PER_STRATEGY = 1  # Number of games per opponent
    ROUNDS_PER_GAME = 20  # Rounds per game
    TEMPERATURE = 0.7  # Sampling temperature (0.0 = greedy)

    # LLM Strategies to instruct
    LLM_STRATEGIES = ['TFT', 'WSLS', 'AllC', 'AllD', 'Grudger']
    
    STRATEGY_PROMPTS = {
        'TFT': "Your strategy is 'Tit-for-Tat': Cooperate on the first round, and then in every subsequent round, do whatever your opponent did in the previous round.",
        'WSLS': "Your strategy is 'Win-Stay, Lose-Shift': Start by cooperating. If you and your opponent both made the same move (both C or both D) in the last round, repeat your move. Otherwise, change your move.",
        'AllC': "Your strategy is 'Always Cooperate': You must choose C (Cooperate) in every single round, regardless of what the opponent does.",
        'AllD': "Your strategy is 'Always Defect': You must choose D (Defect) in every single round, regardless of what the opponent does.",
        'Grudger': "Your strategy is 'Grudger': Start by cooperating. However, if the opponent defects even once, you must defect for the rest of the game."
    }
    
    # Analysis settings
    LAYERS_TO_EXTRACT = [-1, -5, -10]  # Multi-layer probing
    N_CLUSTERS = 6  # Number of clusters for K-means
    USE_PCA_COMPONENTS = 50  # Reduce to 50 dims before clustering for speed
    GAME_POOLING = "mean"  # Pooling for game-level vectors: mean|median|last
    
    # Output settings
    SAVE_PLOTS = True
    OUTPUT_DIR = "/kaggle/working/outputs"
    
    # Memory optimization
    CLEAR_CACHE_FREQUENCY = 10  # Clear GPU cache every N rounds
    BATCH_PROCESS = False  # Process in batches to save memory

# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LLM GAME PLAYER
# ============================================================================

class LLMGamePlayer:
    """Play iterated Prisoner's Dilemma with LLM and extract hidden states"""
    
    def __init__(self, model_name: str = None, use_quantization: bool = True):
        """Initialize LLM player with quantization support"""
        
        if model_name is None:
            model_name = Config.MODEL_NAME
        
        print(f"{'='*60}")
        print(f"Initializing Model: {model_name}")
        print(f"Device: {Config.DEVICE}")
        print(f"Quantization: {use_quantization}")
        print(f"{'='*60}\n")
        
        self.model_name = model_name
        self.device = Config.DEVICE
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for memory efficiency
        if use_quantization and Config.DEVICE == "cuda":
            print("Setting up 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            print("Loading model with quantization (this may take a few minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True,
            )
        else:
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True,
            )
        
        self.model.eval()
        print(f"✓ Model loaded successfully!")
        print(f"  Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB\n")
    
    def create_game_prompt(self, history: List[Tuple[str, str]], 
                          llm_strategy: Optional[str] = None,
                          opponent_last_move: Optional[str] = None) -> str:
        """Create prompt for current game state"""
        
        strategy_instruction = ""
        if llm_strategy and llm_strategy in Config.STRATEGY_PROMPTS:
            strategy_instruction = f"\n{Config.STRATEGY_PROMPTS[llm_strategy]}\n"

        prompt = f"""You are playing the Iterated Prisoner's Dilemma game. In each round:
- Choose C (Cooperate) or D (Defect)
- Payoffs: Both C = 3pts, Both D = 1pt, C vs D = 0pts for C and 5pts for D
{strategy_instruction}
"""
        
        if len(history) > 0:
            prompt += "Game History (last 5 rounds):\n"
            window = history[-5:]
            start_round = len(history) - len(window) + 1
            for i, (my_move, opp_move) in enumerate(window):
                round_num = start_round + i
                prompt += f"Round {round_num}: You={my_move}, Opponent={opp_move}\n"
        
        if opponent_last_move:
            prompt += f"\nOpponent's last move: {opponent_last_move}\n"
        
        prompt += "\nYour move (respond with only C or D): "
        
        return prompt
    
    def get_action_and_hidden_state(self, prompt: str, 
                                   layer_indices: Optional[List[int]] = None):
        """Get LLM's action and extract hidden states (single or multi-layer)"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get hidden states from forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            if layer_indices is None:
                layer_indices = [-1]
            if isinstance(layer_indices, int):
                layer_indices = [layer_indices]

            hidden_vectors = {}
            for layer_index in layer_indices:
                layer_hidden = outputs.hidden_states[layer_index]
                hidden_vectors[layer_index] = layer_hidden[0, -1, :].cpu().float().numpy()
            
            # Generate action
            logits = outputs.logits[0, -1, :]
            
            # Get tokens for C and D
            c_token_id = self.tokenizer.encode('C', add_special_tokens=False)[0]
            d_token_id = self.tokenizer.encode('D', add_special_tokens=False)[0]
            
            # Extract logits for C and D
            c_logit = logits[c_token_id].item()
            d_logit = logits[d_token_id].item()
            
            # Action selection
            if Config.TEMPERATURE <= 0:
                # Greedy selection
                action = 'C' if c_logit > d_logit else 'D'
            else:
                # Stochastic sampling based on temperature
                relevant_logits = torch.tensor([c_logit, d_logit], device=self.device)
                probs = torch.softmax(relevant_logits / Config.TEMPERATURE, dim=0)
                
                # Sample
                choice = torch.multinomial(probs, 1).item()
                action = 'C' if choice == 0 else 'D'
        
        if len(hidden_vectors) == 1:
            return action, list(hidden_vectors.values())[0]
        return action, hidden_vectors
    
    def _get_opponent_action(self, strategy: str, history: List[Tuple[str, str]], 
                           my_last_action: str) -> str:
        """Get opponent's action based on strategy"""
        
        if strategy == 'AllC':
            return 'C'
        elif strategy == 'AllD':
            return 'D'
        elif strategy == 'TFT':
            return 'C' if len(history) == 0 else history[-1][0]
        elif strategy == 'WSLS':
            if len(history) == 0:
                return 'C'
            my_prev, opp_prev = history[-1]
            # Opponent's last payoff determines win-stay/lose-shift
            if opp_prev == 'C' and my_prev == 'C':
                opp_payoff = 3
            elif opp_prev == 'D' and my_prev == 'D':
                opp_payoff = 1
            elif opp_prev == 'D' and my_prev == 'C':
                opp_payoff = 5
            else:
                opp_payoff = 0

            if opp_payoff >= 3:
                return opp_prev
            return 'D' if opp_prev == 'C' else 'C'
        elif strategy == 'Grudger':
            if any(opp == 'D' for _, opp in history):
                return 'D'
            return 'C'
        elif strategy == 'Random':
            return np.random.choice(['C', 'D'])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _calculate_payoff(self, my_action: str, opponent_action: str) -> float:
        """Calculate payoff"""
        if my_action == 'C' and opponent_action == 'C':
            return 3.0
        elif my_action == 'D' and opponent_action == 'D':
            return 1.0
        elif my_action == 'D' and opponent_action == 'C':
            return 5.0
        else:
            return 0.0
    
    def play_game(self, opponent_strategy: str, llm_strategy: str = "TFT", n_rounds: int = 50, 
                 layer_to_extract: Optional[List[int]] = None, game_id: Optional[str] = None) -> List[Dict]:
        """Play a full game and collect data"""
        
        print(f"\n{'='*60}")
        print(f"Playing LLM({llm_strategy}) vs Opponent({opponent_strategy})")
        print(f"{'='*60}")
        
        history = []
        game_data = []
        opponent_last_move = None
        
        for round_num in tqdm(range(n_rounds), desc=f"LLM:{llm_strategy} vs {opponent_strategy}"):
            # Create prompt
            prompt = self.create_game_prompt(history, llm_strategy, opponent_last_move)
            
            # Get action and hidden state
            my_action, hidden_state = self.get_action_and_hidden_state(prompt, layer_to_extract)
            
            # Opponent plays
            opponent_action = self._get_opponent_action(opponent_strategy, history, my_action)
            
            # Calculate payoff
            payoff = self._calculate_payoff(my_action, opponent_action)
            
            # Record
            hidden_states = hidden_state if isinstance(hidden_state, dict) else {layer_to_extract[0] if layer_to_extract is not None and len(layer_to_extract) > 0 else -1: hidden_state}
            game_data.append({
                'game_id': game_id,
                'round': round_num,
                'action': my_action,
                'opponent_action': opponent_action,
                'payoff': payoff,
                'hidden_state': hidden_states[list(hidden_states.keys())[0]],
                'hidden_states': hidden_states,
                'opponent_strategy': opponent_strategy,
                'llm_strategy': llm_strategy
            })
            
            history.append((my_action, opponent_action))
            opponent_last_move = opponent_action
            
            # Clear cache periodically
            if (round_num + 1) % Config.CLEAR_CACHE_FREQUENCY == 0:
                torch.cuda.empty_cache()
        
        total_payoff = sum(d['payoff'] for d in game_data)
        avg_payoff = total_payoff / n_rounds
        coop_rate = sum(1 for d in game_data if d['action'] == 'C') / n_rounds
        
        print(f"  Total payoff: {total_payoff:.1f}")
        print(f"  Avg payoff: {avg_payoff:.2f}")
        print(f"  Cooperation rate: {coop_rate:.1%}")
        
        return game_data

# ============================================================================
# STRATEGY ANALYZER
# ============================================================================

class StrategyAnalyzer:
    """Analyze hidden states to discover strategies"""
    
    def __init__(self):
        self.game_level = False
        self.hidden_states = []
        self.actions = []
        self.opponent_actions = []
        self.payoffs = []
        self.rounds = []
        self.game_ids = []
        self.opponent_strategies = []
        self.llm_strategies = []
        self.coop_rates = []
        self.avg_payoffs = []
        self.tft_similarities = []
    
    def load_data(self, game_data: List[Dict]):
        """Load game data into analyzer"""
        self.game_level = len(game_data) > 0 and 'coop_rate' in game_data[0]
        granularity = "games" if self.game_level else "game rounds"
        print(f"\nLoading {len(game_data)} {granularity}...")
        
        for entry in game_data:
            self.hidden_states.append(entry['hidden_state'])
            self.opponent_strategies.append(entry['opponent_strategy'])
            self.llm_strategies.append(entry.get('llm_strategy', 'Unknown'))
            self.game_ids.append(entry.get('game_id'))
            if self.game_level:
                self.coop_rates.append(entry['coop_rate'])
                self.avg_payoffs.append(entry['avg_payoff'])
                self.tft_similarities.append(entry['tft_similarity'])
                self.rounds.append(entry['rounds'])
            else:
                self.actions.append(entry['action'])
                self.opponent_actions.append(entry['opponent_action'])
                self.payoffs.append(entry['payoff'])
                self.rounds.append(entry['round'])
        
        print(f"✓ Loaded {len(self.hidden_states)} states")
        print(f"  Hidden state dim: {self.hidden_states[0].shape[0]}")
    
    def dimensionality_reduction(self, method='pca', n_components=2, 
                                intermediate_pca=None):
        """Reduce dimensionality for visualization and clustering"""
        
        X = np.array(self.hidden_states)
        print(f"\nOriginal shape: {X.shape}")
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Optional: First reduce to intermediate dimensions for speed
        if intermediate_pca and X.shape[1] > intermediate_pca:
            max_components = min(X.shape[0] - 1, X.shape[1])
            inter_components = min(intermediate_pca, max_components)
            if inter_components < 2:
                inter_components = 2
            print(f"First reducing to {inter_components} dimensions with PCA...")
            pca_intermediate = PCA(n_components=inter_components)
            X_scaled = pca_intermediate.fit_transform(X_scaled)
            print(f"  Explained variance: {pca_intermediate.explained_variance_ratio_.sum():.3f}")
        
        # Final reduction
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            X_reduced = reducer.fit_transform(X_scaled)
            print(f"PCA explained variance: {reducer.explained_variance_ratio_}")
        elif method == 'tsne':
            # Perplexity must be less than n_samples
            n_samples = X_scaled.shape[0]
            perplexity = min(30, n_samples - 1 if n_samples > 1 else 1)
            print(f"Running t-SNE with perplexity={perplexity}...")
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_jobs=-1)
            X_reduced = reducer.fit_transform(X_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return X_reduced, X_scaled
    
    def find_optimal_clusters(self, X_scaled, k_range=range(2, 10)):
        """Find optimal number of clusters using silhouette score"""
        
        print("\nFinding optimal number of clusters...")
        silhouette_scores = []
        
        for k in tqdm(k_range, desc="Testing K values"):
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            print(f"  K={k}: Silhouette = {score:.3f}")
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n✓ Optimal K: {optimal_k}")
        
        return optimal_k, silhouette_scores
    
    def cluster_analysis(self, X_scaled, n_clusters=5):
        """Perform clustering"""
        print(f"\nClustering with K={n_clusters}...")
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)
        
        print(f"✓ Clustering complete")
        return labels
    
    def analyze_cluster_behaviors(self, cluster_labels):
        """Analyze behavior of each cluster"""
        
        print(f"\n{'='*60}")
        print("CLUSTER BEHAVIOR ANALYSIS")
        print(f"{'='*60}")
        
        n_clusters = len(np.unique(cluster_labels))
        cluster_stats = {}
        
        for label in range(n_clusters):
            indices = [i for i, l in enumerate(cluster_labels) if l == label]
            sample_count = len(indices)
            
            rounds = [self.rounds[i] for i in indices]
            opp_strategies = [self.opponent_strategies[i] for i in indices]
            llm_strategies = [self.llm_strategies[i] for i in indices]
            
            # Calculate statistics
            if self.game_level:
                coop_rate = float(np.mean([self.coop_rates[i] for i in indices]))
                avg_payoff = float(np.mean([self.avg_payoffs[i] for i in indices]))
                tft_similarity = float(np.mean([self.tft_similarities[i] for i in indices]))
                actions = []
                opp_actions = []
            else:
                actions = [self.actions[i] for i in indices]
                opp_actions = [self.opponent_actions[i] for i in indices]
                payoffs = [self.payoffs[i] for i in indices]
                coop_rate = sum(1 for a in actions if a == 'C') / len(actions)
                avg_payoff = np.mean(payoffs)
                
                # TFT similarity computed within each game in chronological order
                tft_matches = 0
                tft_total = 0
                game_ids = {self.game_ids[i] for i in indices}
                for gid in game_ids:
                    game_indices = [i for i in indices if self.game_ids[i] == gid]
                    game_indices.sort(key=lambda i: self.rounds[i])
                    for j in range(1, len(game_indices)):
                        idx_curr = game_indices[j]
                        idx_prev = game_indices[j - 1]
                        if self.rounds[idx_curr] == self.rounds[idx_prev] + 1:
                            tft_total += 1
                            if self.actions[idx_curr] == self.opponent_actions[idx_prev]:
                                tft_matches += 1
                tft_similarity = tft_matches / tft_total if tft_total > 0 else 0
            
            # Strategy distribution
            opp_strategy_dist = Counter(opp_strategies)
            llm_strategy_dist = Counter(llm_strategies)
            
            cluster_stats[label] = {
                'size': sample_count,
                'coop_rate': coop_rate,
                'avg_payoff': avg_payoff,
                'tft_similarity': tft_similarity,
                'opp_strategy_dist': opp_strategy_dist,
                'llm_strategy_dist': llm_strategy_dist,
                'actions': actions,
                'opponent_actions': opp_actions
            }
            
            print(f"\nCluster {label} ({sample_count} samples):")
            print(f"  Cooperation rate: {coop_rate:.1%}")
            print(f"  Avg payoff: {avg_payoff:.2f}")
            print(f"  TFT similarity: {tft_similarity:.1%}")
            print(f"  Dominant instructed strategy: {llm_strategy_dist.most_common(1)[0][0]}")
            print(f"  Opponent strategies: {dict(opp_strategy_dist.most_common(3))}")
            
        return cluster_stats
    
    def visualize_results(self, X_reduced_pca, X_reduced_tsne, cluster_labels, 
                         cluster_stats, silhouette_scores=None, k_range=None):
        """Create comprehensive visualizations"""
        
        print("\nGenerating visualizations...")
        
        # Figure 1: Clustering results
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # PCA colored by cluster
        scatter1 = axes[0, 0].scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1],
                                     c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
        axes[0, 0].set_title('PCA - Colored by Cluster', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
        
        # PCA colored by Instructed LLM Strategy
        unique_strategies = sorted(list(set(self.llm_strategies)))
        strategy_to_idx = {s: i for i, s in enumerate(unique_strategies)}
        strategy_colors = [strategy_to_idx[s] for s in self.llm_strategies]
        
        scatter2 = axes[0, 1].scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1],
                                     c=strategy_colors, cmap='tab10', alpha=0.6, s=30)
        axes[0, 1].set_title('PCA - Colored by Instructed LLM Strategy',
                             fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('PC1')
        axes[0, 1].set_ylabel('PC2')
        cbar = plt.colorbar(scatter2, ax=axes[0, 1])
        cbar.set_ticks(range(len(unique_strategies)))
        cbar.ax.set_yticklabels(unique_strategies)
        cbar.set_label('Instructed Strategy')
        
        # t-SNE colored by cluster
        scatter3 = axes[1, 0].scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1],
                                     c=cluster_labels, cmap='viridis', alpha=0.6, s=30)
        axes[1, 0].set_title('t-SNE - Colored by Cluster', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster')
        
        # Silhouette scores
        if silhouette_scores and k_range:
            axes[1, 1].plot(list(k_range), silhouette_scores, 'bo-', linewidth=2, markersize=8)
            axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
            axes[1, 1].set_ylabel('Silhouette Score', fontsize=12)
            axes[1, 1].set_title('Optimal K Selection', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            optimal_k = k_range[np.argmax(silhouette_scores)]
            axes[1, 1].axvline(optimal_k, color='r', linestyle='--', linewidth=2, 
                              label=f'Optimal K={optimal_k}')
            axes[1, 1].legend()
        
        plt.tight_layout()
        if Config.SAVE_PLOTS:
            plt.savefig(f"{Config.OUTPUT_DIR}/clustering_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Figure 2: Cluster statistics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Cooperation rate by cluster
        clusters = sorted(cluster_stats.keys())
        coop_rates = [cluster_stats[c]['coop_rate'] for c in clusters]
        axes[0, 0].bar(clusters, coop_rates, color='skyblue', edgecolor='navy', linewidth=1.5)
        axes[0, 0].set_xlabel('Cluster', fontsize=12)
        axes[0, 0].set_ylabel('Cooperation Rate', fontsize=12)
        axes[0, 0].set_title('Cooperation Rate by Cluster', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Average payoff by cluster
        avg_payoffs = [cluster_stats[c]['avg_payoff'] for c in clusters]
        axes[0, 1].bar(clusters, avg_payoffs, color='lightcoral', edgecolor='darkred', linewidth=1.5)
        axes[0, 1].set_xlabel('Cluster', fontsize=12)
        axes[0, 1].set_ylabel('Average Payoff', fontsize=12)
        axes[0, 1].set_title('Average Payoff by Cluster', fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Cluster sizes
        sizes = [cluster_stats[c]['size'] for c in clusters]
        axes[1, 0].bar(clusters, sizes, color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
        axes[1, 0].set_xlabel('Cluster', fontsize=12)
        axes[1, 0].set_ylabel('Number of Samples', fontsize=12)
        axes[1, 0].set_title('Cluster Sizes', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Heatmap: Cluster vs Instructed LLM Strategy
        llm_strategies = Config.LLM_STRATEGIES if hasattr(Config, 'LLM_STRATEGIES') else unique_strategies
        heatmap_data = np.zeros((len(clusters), len(llm_strategies)))
        
        for i, cluster in enumerate(clusters):
            for j, strategy in enumerate(llm_strategies):
                count = cluster_stats[cluster]['llm_strategy_dist'].get(strategy, 0)
                heatmap_data[i, j] = count / cluster_stats[cluster]['size']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu',
                   xticklabels=llm_strategies, yticklabels=[f'C{c}' for c in clusters],
                   ax=axes[1, 1], cbar_kws={'label': 'Proportion'})
        axes[1, 1].set_title('Cluster Composition by Instructed Strategy', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Instructed LLM Strategy', fontsize=12)
        axes[1, 1].set_ylabel('Cluster', fontsize=12)
        
        plt.tight_layout()
        if Config.SAVE_PLOTS:
            plt.savefig(f"{Config.OUTPUT_DIR}/cluster_statistics.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizations complete")

# ============================================================================
# GAME-LEVEL AGGREGATION
# ============================================================================

def aggregate_game_rounds(all_game_data: List[Dict], pooling: str = "mean") -> List[Dict]:
    """Aggregate round-level data into game-level vectors."""

    grouped = defaultdict(list)
    for entry in all_game_data:
        grouped[entry['game_id']].append(entry)

    game_level_data = []
    for game_id, entries in grouped.items():
        hidden_stack = np.stack([e['hidden_state'] for e in entries], axis=0)
        if pooling == "median":
            pooled = np.median(hidden_stack, axis=0)
        elif pooling == "last":
            pooled = hidden_stack[-1]
        else:
            pooled = hidden_stack.mean(axis=0)

        actions = [e['action'] for e in entries]
        opp_actions = [e['opponent_action'] for e in entries]
        payoffs = [e['payoff'] for e in entries]

        coop_rate = sum(1 for a in actions if a == 'C') / len(actions)
        avg_payoff = float(np.mean(payoffs))

        tft_matches = 0
        for i in range(1, len(actions)):
            if actions[i] == opp_actions[i - 1]:
                tft_matches += 1
        tft_similarity = tft_matches / (len(actions) - 1) if len(actions) > 1 else 0

        opponent_strategy = entries[0]['opponent_strategy'] if entries else "Unknown"
        llm_strategy = entries[0].get('llm_strategy', "Unknown")
        
        game_level_data.append({
            'game_id': game_id,
            'opponent_strategy': opponent_strategy,
            'llm_strategy': llm_strategy,
            'rounds': len(entries),
            'coop_rate': coop_rate,
            'avg_payoff': avg_payoff,
            'tft_similarity': tft_similarity,
            'hidden_state': pooled,
        })

    return game_level_data

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def _extract_layer_data(all_game_data: List[Dict], layer_index: int) -> List[Dict]:
    """Extract a single layer's hidden states into a flat dataset."""
    layer_data = []
    for entry in all_game_data:
        if 'hidden_states' not in entry:
            layer_data.append(entry)
            continue
        if layer_index not in entry['hidden_states']:
            continue
        entry_copy = entry.copy()
        entry_copy['hidden_state'] = entry['hidden_states'][layer_index]
        layer_data.append(entry_copy)
    return layer_data


def run_full_experiment():
    """Run complete experiment pipeline"""
    
    print("\n" + "="*60)
    print("LLM STRATEGY DISCOVERY EXPERIMENT")
    print("="*60)
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Device: {Config.DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"LLM Strategies: {Config.LLM_STRATEGIES}")
    print(f"Opponent Strategies: {Config.OPPONENT_STRATEGIES}")
    print(f"Games per pair: {Config.GAMES_PER_STRATEGY}")
    print(f"Rounds per game: {Config.ROUNDS_PER_GAME}")
    print(f"Layers to extract: {Config.LAYERS_TO_EXTRACT}")
    print("="*60 + "\n")
    
    # Step 1: Initialize player
    player = LLMGamePlayer(
        model_name=Config.MODEL_NAME,
        use_quantization=Config.USE_QUANTIZATION
    )
    
    # Step 2: Collect game data
    all_game_data = []
    
    for llm_strategy in Config.LLM_STRATEGIES:
        print(f"\n--- TESTING LLM STRATEGY: {llm_strategy} ---")
        for opponent_strategy in Config.OPPONENT_STRATEGIES:
            for game_idx in range(Config.GAMES_PER_STRATEGY):
                game_id = f"LLM_{llm_strategy}_vs_{opponent_strategy}_{game_idx}"
                game_data = player.play_game(
                    opponent_strategy=opponent_strategy,
                    llm_strategy=llm_strategy,
                    n_rounds=Config.ROUNDS_PER_GAME,
                    layer_to_extract=Config.LAYERS_TO_EXTRACT,
                    game_id=game_id
                )
                all_game_data.extend(game_data)
            
            # Clear cache between opponent strategies for the same LLM strategy
            torch.cuda.empty_cache()
        
        # Clear cache between LLM strategies
        torch.cuda.empty_cache()
    
    # Save raw data (all layers)
    print(f"\n{'='*60}")
    print("Saving game data...")
    
    with open(f"{Config.OUTPUT_DIR}/game_data_all_layers.json", 'w') as f:
        json.dump(all_game_data, f, cls=NumpyEncoder)
    print(f"✓ Saved {len(all_game_data)} game rounds (all layers)")
    
    # Basic statistics
    df = pd.DataFrame([{k: v for k, v in d.items() if k != 'hidden_state' and k != 'hidden_states'} for d in all_game_data])
    print(f"\n{'='*60}")
    print("GAME STATISTICS")
    print(f"{'='*60}")
    print(f"Total rounds: {len(all_game_data)}")
    print(f"Overall cooperation rate: {(df['action'] == 'C').mean():.1%}")
    print(f"Overall average payoff: {df['payoff'].mean():.2f}")
    
    print("\nBy LLM strategy:")
    print(df.groupby('llm_strategy').agg({
        'payoff': 'mean',
        'action': lambda x: (x == 'C').mean()
    }).round(3))

    print("\nBy opponent strategy:")
    print(df.groupby('opponent_strategy').agg({
        'payoff': 'mean',
        'action': lambda x: (x == 'C').mean()
    }).round(3))
    
    # Step 3: Analyze hidden states per layer
    for layer_index in Config.LAYERS_TO_EXTRACT:
        layer_dir = f"{Config.OUTPUT_DIR}/layer_{layer_index}"
        os.makedirs(layer_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"ANALYZING HIDDEN STATES (layer {layer_index})")
        print(f"{'='*60}")

        layer_game_data = _extract_layer_data(all_game_data, layer_index)
        game_level_data = aggregate_game_rounds(layer_game_data, pooling=Config.GAME_POOLING)

        with open(f"{layer_dir}/game_data_layer_{layer_index}.json", 'w') as f:
            json.dump(layer_game_data, f, cls=NumpyEncoder)
        print(f"✓ Saved {len(layer_game_data)} game rounds (layer {layer_index})")

        with open(f"{layer_dir}/game_level_data_layer_{layer_index}.json", 'w') as f:
            json.dump(game_level_data, f, cls=NumpyEncoder)
        print(f"✓ Saved {len(game_level_data)} games (aggregated, layer {layer_index})")

        if len(game_level_data) < 2:
            print("Not enough games for PCA/t-SNE/clustering. Skipping analysis.")
            continue

        analyzer = StrategyAnalyzer()
        analyzer.load_data(game_level_data)

        print("\nPerforming PCA...")
        X_reduced_pca, X_scaled = analyzer.dimensionality_reduction(
            method='pca',
            n_components=2,
            intermediate_pca=Config.USE_PCA_COMPONENTS
        )

        print("\nPerforming t-SNE...")
        X_reduced_tsne, _ = analyzer.dimensionality_reduction(
            method='tsne',
            n_components=2,
            intermediate_pca=Config.USE_PCA_COMPONENTS
        )

        k_max = max(2, min(10, len(game_level_data)))
        k_range = range(2, k_max + 1)
        optimal_k, silhouette_scores = analyzer.find_optimal_clusters(
            X_scaled,
            k_range=k_range
        )

        n_clusters = Config.N_CLUSTERS if Config.N_CLUSTERS else optimal_k
        n_clusters = min(n_clusters, len(game_level_data))

        cluster_labels = analyzer.cluster_analysis(X_scaled, n_clusters=n_clusters)
        cluster_stats = analyzer.analyze_cluster_behaviors(cluster_labels)

        # Temporarily redirect outputs
        original_output_dir = Config.OUTPUT_DIR
        Config.OUTPUT_DIR = layer_dir
        analyzer.visualize_results(
            X_reduced_pca,
            X_reduced_tsne,
            cluster_labels,
            cluster_stats,
            silhouette_scores,
            k_range=k_range
        )
        Config.OUTPUT_DIR = original_output_dir

        print(f"\n{'='*60}")
        print("GENERATING REPORT")
        print(f"{'='*60}")

        report_path = f"{layer_dir}/analysis_report_layer_{layer_index}.txt"
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LLM STRATEGY ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Model: {Config.MODEL_NAME}\n")
            f.write(f"Layer: {layer_index}\n")
            f.write(f"Total rounds: {len(layer_game_data)}\n")
            f.write(f"Total games: {len(game_level_data)}\n")
            f.write(f"Pooling: {Config.GAME_POOLING}\n")
            f.write(f"Hidden state dimension: {game_level_data[0]['hidden_state'].shape[0]}\n")
            f.write(f"Number of clusters: {n_clusters}\n\n")

            f.write("Cluster Analysis:\n")
            f.write("-" * 60 + "\n")
            for label, stats in cluster_stats.items():
                f.write(f"\nCluster {label}:\n")
                f.write(f"  Size: {stats['size']}\n")
                f.write(f"  Cooperation rate: {stats['coop_rate']:.1%}\n")
                f.write(f"  Average payoff: {stats['avg_payoff']:.2f}\n")
                f.write(f"  TFT similarity: {stats['tft_similarity']:.1%}\n")
                f.write(f"  Dominant instructed strategy: {stats['llm_strategy_dist'].most_common(1)[0][0]}\n")
                f.write(f"  Top opponent strategies: {dict(stats['opp_strategy_dist'].most_common(3))}\n")

        print(f"✓ Report saved to {report_path}")

        results = {
            'model': Config.MODEL_NAME,
            'layer': layer_index,
            'n_clusters': n_clusters,
            'optimal_k': optimal_k,
            'cluster_stats': {
                str(k): {
                    'size': v['size'],
                    'coop_rate': v['coop_rate'],
                    'avg_payoff': v['avg_payoff'],
                    'tft_similarity': v['tft_similarity']
                } for k, v in cluster_stats.items()
            }
        }

        with open(f"{layer_dir}/results_layer_{layer_index}.json", 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

        print(f"\n{'='*60}")
        print("EXPERIMENT COMPLETE!")
        print(f"{'='*60}")
        print(f"Results saved to: {layer_dir}")
        print(f"  - game_data_layer_{layer_index}.json: Raw game data")
        print(f"  - game_level_data_layer_{layer_index}.json: Aggregated game data")
        print(f"  - clustering_analysis.png: Visualization")
        print(f"  - cluster_statistics.png: Statistics plots")
        print(f"  - analysis_report_layer_{layer_index}.txt: Text report")
        print(f"  - results_layer_{layer_index}.json: Summary results")

    return all_game_data, None, None

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available, running on CPU (will be slow)")
    
    # Run experiment
    try:
        all_game_data, cluster_labels, cluster_stats = run_full_experiment()
        print("\n✓ All tasks completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nFinal GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")