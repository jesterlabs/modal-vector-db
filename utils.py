import json
import uuid
import numpy as np
from vicinity.datatypes import QueryResult
from vicinity.utils import Metric, normalize
import numpy.typing as npt


def json_to_uuid(json_data, namespace=uuid.NAMESPACE_DNS):
    """Convert JSON data to a deterministic UUID."""
    json_str = json.dumps(json_data, sort_keys=True)  # Ensure consistent ordering
    return uuid.uuid5(namespace, json_str)

def build_where_clause(filters: dict) -> str:
    """
    Convert a dictionary of filters to a DuckDB WHERE clause.
    
    Supports operators:
    - eq: equals (default when no operator specified)
    - gt: greater than
    - gte: greater than or equal
    - lt: less than
    - lte: less than or equal
    - in: list membership
    - between: range inclusive
    
    Example:
    filters = {
        'type': 'fire',                    # equals
        'attack__gt': 80,                  # greater than
        'defense__lte': 100,               # less than or equal
        'name__in': ['charizard', 'moltres'], # in list
        'speed__between': [50, 100]        # between range
    }
    """
    operators = {
        'gt': '>',
        'gte': '>=',
        'lt': '<',
        'lte': '<=',
        'in': 'IN',
        'between': 'BETWEEN'
    }
    
    conditions = []
    
    for key, value in filters.items():
        parts = key.split('__')
        field = parts[0]
        op = parts[1] if len(parts) > 1 else 'eq'
        
        if op == 'eq':
            conditions.append(f"{field} = '{value}'" if isinstance(value, str) else f"{field} = {value}")
        elif op == 'in':
            formatted_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
            conditions.append(f"{field} IN ({', '.join(formatted_values)})")
        elif op == 'between':
            conditions.append(f"{field} BETWEEN {value[0]} AND {value[1]}")
        elif op in operators:
            value_str = f"'{value}'" if isinstance(value, str) else str(value)
            conditions.append(f"{field} {operators[op]} {value_str}")
            
    return ' AND '.join(conditions) if conditions else '1=1'

def numpy_similarity_calculation(query_vectors: npt.NDArray, vectors: npt.NDArray, metric: Metric) -> list[QueryResult]:
    """Calculate similarity scores between query vectors and stored vectors using specified metric.
    
    Args:
        query_vectors: Query vectors of shape (n_queries, dim)
        vectors: Database vectors of shape (n_vectors, dim)
        metric: Similarity metric to use
        
    Returns:
        List of (index, score) tuples for each query vector
    """
    if metric == Metric.COSINE:
        query_vectors = normalize(query_vectors)
        vectors = normalize(vectors)
        scores = np.dot(query_vectors, vectors.T)
        
    elif metric == Metric.EUCLIDEAN:
        scores = np.sqrt(((query_vectors[:, None] - vectors) ** 2).sum(axis=2))
        scores = -scores
        
    elif metric == Metric.MANHATTAN:
        scores = np.abs(query_vectors[:, None] - vectors).sum(axis=2)
        scores = -scores
        
    elif metric == Metric.INNER_PRODUCT:
        scores = np.dot(query_vectors, vectors.T)
        
    elif metric == Metric.L2_SQUARED:
        scores = ((query_vectors[:, None] - vectors) ** 2).sum(axis=2)
        scores = -scores
        
    elif metric == Metric.HAMMING:
        scores = np.count_nonzero(query_vectors[:, None] != vectors, axis=2)
        scores = -scores
        
    elif metric == Metric.TANIMOTO:
        intersection = np.dot(query_vectors, vectors.T)
        query_sum = (query_vectors ** 2).sum(axis=1)[:, None]
        vector_sum = (vectors ** 2).sum(axis=1)
        union = query_sum + vector_sum - intersection
        scores = intersection / union
        
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    results = []
    for query_scores in scores:
        query_results = list(enumerate(query_scores))
        results.append(query_results)
        
    return results


def numpy_similarity_threshold_calculation(query_vectors: npt.NDArray, vectors: npt.NDArray, metric: Metric, threshold: float) -> list[QueryResult]:
    """Calculate similarity scores between query vectors and stored vectors, returning only matches above threshold.
    
    Args:
        query_vectors: Query vectors of shape (n_queries, dim)
        vectors: Database vectors of shape (n_vectors, dim)
        metric: Similarity metric to use
        threshold: Minimum similarity score threshold
        
    Returns:
        List of (index, score) tuples for each query vector where score >= threshold
    """
    all_results = numpy_similarity_calculation(query_vectors, vectors, metric)
    
    threshold_results = []
    for query_results in all_results:
        filtered_results = [
            (idx, score) for idx, score in query_results 
            if score >= threshold
        ]
        threshold_results.append(filtered_results)
        
    return threshold_results