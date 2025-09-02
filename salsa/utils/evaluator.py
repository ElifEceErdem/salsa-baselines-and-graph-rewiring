import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
from tqdm import tqdm


class Evaluator:
    def __init__(self, k=10):
        """
        Evaluator for link prediction tasks.

        Args:
            k (int): Number of top recommendations to consider for metrics like Precision@K, Recall@K, and NDCG@K.
        """
        self.k = k

    # Precision@K
    def precision_at_k(self, y_true, y_scores, node_indices):
        node_groups = self._prepare_node_groups(node_indices, y_true, y_scores)
        precision_per_node = []
        for src, edges in node_groups.items():
            edges.sort(key=lambda x: x[1], reverse=True)
            top_k_edges = edges[:self.k]
            relevant_items = sum(label for label, _ in top_k_edges)
            precision = relevant_items / self.k
            precision_per_node.append(precision)
        return np.mean(precision_per_node)

    # Recall@K
    def recall_at_k(self, y_true, y_scores, node_indices):
        node_groups = self._prepare_node_groups(node_indices, y_true, y_scores)
        recall_per_node = []
        for src, edges in node_groups.items():
            edges.sort(key=lambda x: x[1], reverse=True)
            top_k_edges = edges[:self.k]
            relevant_items_in_top_k = sum(label for label, _ in top_k_edges)
            total_relevant_items = sum(label for label, _ in edges)
            recall = relevant_items_in_top_k / total_relevant_items if total_relevant_items > 0 else 0
            recall_per_node.append(recall)
        return np.mean(recall_per_node)

    # NDCG@K
    def ndcg_at_k(self, y_true, y_scores, node_indices):
        node_groups = self._prepare_node_groups(node_indices, y_true, y_scores)
        ndcg_per_node = []
        for src, edges in node_groups.items():
            edges.sort(key=lambda x: x[1], reverse=True)
            top_k_edges = edges[:self.k]
            dcg = sum((label / np.log2(idx + 2)) for idx, (label, _) in enumerate(top_k_edges))
            ideal_edges = sorted(edges, key=lambda x: x[0], reverse=True)[:self.k]
            idcg = sum((label / np.log2(idx + 2)) for idx, (label, _) in enumerate(ideal_edges))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_per_node.append(ndcg)
        return np.mean(ndcg_per_node)

    def compute_auc_and_map(self, y_true, y_scores):
        if len(y_true) == 0:
            return 0, 0
        
        # Check if there are at least two classes in y_true
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            # If only one class is present, AUC is not defined
            auc = 0
        else:
            auc = roc_auc_score(y_true, y_scores)
        
        # MAP can still be computed even if there's only one class
        map_score = average_precision_score(y_true, y_scores)
        
        return auc, map_score

    # All metrics
    def compute_metrics(self, y_true, y_scores, node_indices):
        precision = self.precision_at_k(y_true, y_scores, node_indices)
        recall = self.recall_at_k(y_true, y_scores, node_indices)
        ndcg = self.ndcg_at_k(y_true, y_scores, node_indices)
        auc, map_score = self.compute_auc_and_map(y_true, y_scores)
        return {
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "auc": auc,
            "map": map_score,
        }
    
    def evaluate_all_predictions(self, predicted_links, test_edges, k=None):
        if k is None:
            k = self.k
        precision_list, recall_list, ndcg_list, ap_list, y_true, y_scores = [], [], [], [], [], []
        
        update_interval = 10000  # Update tqdm every 100k nodes
        progress_bar = tqdm(total=len(predicted_links), desc="Evaluation on Predicted Links", position=0)

        for i, node in enumerate(predicted_links.keys(), start=1):
            relevant_items = test_edges[test_edges[:, 0] == node][:, 1]
            recommended_items = predicted_links[node]

            if len(relevant_items) > 0:
                y_true_node = [1 if v in relevant_items else 0 for v in recommended_items]
                y_scores_node = list(range(len(recommended_items), 0, -1))
                y_true.extend(y_true_node)
                y_scores.extend(y_scores_node)

                precision, recall, ndcg, ap_k = self.evaluate_node_predictions(relevant_items, recommended_items, k)
                precision_list.append(precision)
                recall_list.append(recall)
                ndcg_list.append(ndcg)
                ap_list.append(ap_k)
            else:
                precision_list.append(0)
                recall_list.append(0)
                ndcg_list.append(0)
                ap_list.append(0)

            # Update progress bar every 100k nodes
            if i % update_interval == 0:
                progress_bar.update(update_interval)

        # Ensure progress bar is fully updated
        progress_bar.update(len(predicted_links) % update_interval)
        progress_bar.close()

        print(f"mean_precision: {np.mean(precision_list)}")
        print(f"mean_recall: {np.mean(recall_list)}")
        print(f"mean_ndcg: {np.mean(ndcg_list)}")
        print(f"map_at_k: {np.mean(ap_list)}")

        auc, map_score = self.compute_auc_and_map(y_true, y_scores)
        print(f"auc: ", auc)
        print(f"map_score: ", map_score)

        return {
            "mean_precision": np.mean(precision_list),
            "mean_recall": np.mean(recall_list),
            "mean_ndcg": np.mean(ndcg_list),
            "map_at_k": np.mean(ap_list),
            "auc": auc,
            "map": map_score,
        }

    # Helper methods
    def _prepare_node_groups(self, node_indices, y_true, y_scores):
        node_groups = defaultdict(list)
        for i, (src, tgt) in enumerate(node_indices):
            node_groups[src].append((y_true[i], y_scores[i]))
        return node_groups
    
    def evaluate_node_predictions(self, relevant_items, recommended_items, k):
        relevant_set = set(relevant_items)
        recommended_set = set(recommended_items[:k])
        intersection = relevant_set.intersection(recommended_set)

        precision = len(intersection) / len(recommended_set) if recommended_set else 0
        recall = len(intersection) / len(relevant_set) if relevant_set else 0
        ndcg = self._compute_ndcg(relevant_items, recommended_items, k)

        # Compute AP@K
        ap_k = 0.0
        num_hits = 0
        for i, item in enumerate(recommended_items[:k], start=1):
            if item in relevant_set:
                num_hits += 1
                ap_k += num_hits / i
        ap_k = ap_k / min(len(relevant_set), k) if relevant_set else 0

        return precision, recall, ndcg, ap_k

    def _compute_ndcg(self, relevant_items, recommended_items, k):
        dcg = 0
        idcg = sum([1.0 / (i + 1) for i in range(min(len(relevant_items), k))])
        for i, rec in enumerate(recommended_items[:k]):
            if rec in relevant_items:
                dcg += 1.0 / (i + 1)
        return dcg / idcg if idcg > 0 else 0
