import numpy as np
import math

# Local structural similarity measures
class LocalStructuralSimilarity:
    def __init__(self, adj_csr, out_neighbors, in_neighbors):
        self.adj_csr = adj_csr
        self.out_neighbors = out_neighbors
        self.in_neighbors = in_neighbors

    def common_neighbors(self, u, v):
        neighbors_u = set(self.adj_csr[u])
        neighbors_v = set(self.adj_csr[v])
        return len(neighbors_u.intersection(neighbors_v))
    
    def common_neighbors_directed(self, u, v):
        out_u = self.out_neighbors.get(u, set())
        in_v = self.in_neighbors.get(v, set())
        return len(out_u.intersection(in_v))

    def adamic_adar_similarity(self, u, v):
        neighbors_u = set(self.adj_csr[u])
        neighbors_v = set(self.adj_csr[v])
        common_neighbors = neighbors_u.intersection(neighbors_v)
        if not common_neighbors:
            return 0
        adamic_adar_score = sum(1 / math.log(len(self.adj_csr[t])) for t in common_neighbors if len(self.adj_csr[t]) > 1)
        return adamic_adar_score

    def directed_adamic_adar_similarity(self, u, v):
        out_u = self.out_neighbors.get(u, set())
        in_v = self.in_neighbors.get(v, set())

        # Find common neighbors
        common_neighbors = out_u.intersection(in_v)
        
        if not common_neighbors:  # If no common neighbors, similarity is 0
            return (0,0)
        
        # Compute Adamic-Adar similarity (sum contributions of common neighbors)
        adamic_adar_score_out = 0
        adamic_adar_score_in = 0

        for t in (common_neighbors):
            # Get the degree of common neighbor t (sum of out and in degrees for t)
            #degree_t = len(set(adj_csr[t]) + len(adj_csr[:, t])

            out_degree_t = len(self.out_neighbors.get(t, set()))
            in_degree_t = len(self.in_neighbors.get(t, set()))
            
            if out_degree_t > 1:
                adamic_adar_score_out += 1 / math.log(out_degree_t)
            if in_degree_t > 1:
                adamic_adar_score_in += 1 / math.log(in_degree_t)
        
        return adamic_adar_score_out, adamic_adar_score_in

    def resource_allocation_similarity(self, u, v):
        neighbors_u = set(self.adj_csr[u])
        neighbors_v = set(self.adj_csr[v])
        common_neighbors = neighbors_u.intersection(neighbors_v)
        if not common_neighbors:
            return 0
        ra_score = sum(1 / len(self.adj_csr[t]) for t in common_neighbors if len(self.adj_csr[t]) > 0)
        return ra_score
    
    def resource_allocation_directed(self, u, v):
        # Get the neighbors of u and v
        
        out_u = self.out_neighbors.get(u, set())
        in_v = self.in_neighbors.get(v, set())
        common_neighbors = out_u.intersection(in_v)
        
        if not common_neighbors:  # If no common neighbors, similarity is 0
            return (0, 0)
        
        out_degree_u = len(out_u)  # Degree of node u (|Γ(u)|)
        in_degree_u = len(in_v)

        # Compute Resource Allocation similarity (sum contributions of common neighbors)
        ra_score_in = 0
        ra_score_out = 0

        for t in common_neighbors:
            out_degree_t = len(self.out_neighbors.get(t, set()))  # Get the degree of common neighbor t
            in_degree_t = len(self.in_neighbors.get(t, set()))
            
            if out_degree_t > 0:  # Avoid division by zero
                ra_score_out += 1 / out_degree_t
            if in_degree_t > 0:  # Avoid division by zero
                ra_score_in += 1 / in_degree_t

            if out_degree_u == 0:
                ra_score_out = 0
            elif in_degree_u == 0:
                ra_score_in = 0
            
            else:
                ra_score_out = ra_score_out/out_degree_u
                ra_score_in = ra_score_in/in_degree_u
            
        return ra_score_out, ra_score_in

    def jaccard_similarity(self, u, v):
        neighbors_u = set(self.adj_csr[u])
        neighbors_v = set(self.adj_csr[v])
        intersection = len(neighbors_u.intersection(neighbors_v))
        union = len(neighbors_u.union(neighbors_v))
        return intersection / union if union > 0 else 0

    def directed_jaccard_similarity(self, u, v):
        out_u = self.out_neighbors.get(u, set())
        in_v = self.in_neighbors.get(v, set())
        if len(out_u.union(in_v)) == 0:  # Avoid division by zero
            return 0

        return len(out_u.intersection(in_v)) / len(out_u.union(in_v))

    def cosine_similarity(self, u, v):
        # Get the neighbors and weights of u and v
        neighbors_u = set(self.adj_csr[u])
        neighbors_v = set(self.adj_csr[v])
        
        # Find common neighbors
        common_neighbors = neighbors_u.intersection(neighbors_v)
        
        if not common_neighbors:  # If no common neighbors, similarity is 0
            return 0
        
        weights_u = {neighbor: 1 for neighbor in neighbors_u}
        weights_v = {neighbor: 1 for neighbor in neighbors_v}
        
        # Compute the dot product (numerator)
        numerator = sum(weights_u[t] * weights_v[t] for t in common_neighbors)
        
        # Compute the magnitude of the weight vector for u (denominator part 1)
        magnitude_u = np.sqrt(sum(weight ** 2 for weight in weights_u.values()))
        
        # Compute the magnitude of the weight vector for v (denominator part 2)
        magnitude_v = np.sqrt(sum(weight ** 2 for weight in weights_v.values()))
        
        if magnitude_u == 0 or magnitude_v == 0:  # Avoid division by zero
            return 0
        
        # Calculate cosine similarity
        cosine_sim = numerator / (magnitude_u * magnitude_v)
        
        return cosine_sim
