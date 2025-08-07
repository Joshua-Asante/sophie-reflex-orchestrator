"""
Core Graph Utilities Module

Provides graph algorithms including topological sorting.
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def topological_sort(graph: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Perform topological sort on a directed acyclic graph.
    
    Args:
        graph: Dictionary mapping node names to their dependencies
        
    Returns:
        List of node names in topological order
    """
    # Kahn's algorithm for topological sorting
    in_degree = {}
    for node in graph:
        in_degree[node] = len(graph[node].get('depends_on', []))
    
    # Find nodes with no incoming edges
    queue = [node for node in graph if in_degree[node] == 0]
    result = []
    
    while queue:
        # Remove a node with no incoming edges
        current = queue.pop(0)
        result.append(current)
        
        # Reduce in-degree of all neighbors
        for node in graph:
            if current in graph[node].get('depends_on', []):
                in_degree[node] -= 1
                if in_degree[node] == 0:
                    queue.append(node)
    
    # Check for cycles
    if len(result) != len(graph):
        raise ValueError("Graph contains cycles")
    
    return result


def has_cycle(graph: Dict[str, Dict[str, Any]]) -> bool:
    """
    Check if a graph contains cycles using DFS.
    
    Args:
        graph: Dictionary mapping node names to their dependencies
        
    Returns:
        True if graph contains cycles, False otherwise
    """
    visited = set()
    rec_stack = set()
    
    def dfs(node: str) -> bool:
        if node in rec_stack:
            return True
        if node in visited:
            return False
        
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph[node].get('depends_on', []):
            if neighbor in graph and dfs(neighbor):
                return True
        
        rec_stack.remove(node)
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    
    return False 