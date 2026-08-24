//! Semantic knowledge graph indexing across workspaces, symbols, and derivations (WS16).

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphIndexError {
    #[error("Cycle detected in semantic dependency graph: {0}")]
    CycleDetected(String),
    #[error("Node not found: {0}")]
    NodeNotFound(String),
}

/// Node kind in the semantic knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    Workspace,
    Symbol,
    Theorem,
    Derivation,
}

/// A node in the semantic knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub kind: NodeKind,
    pub metadata: HashMap<String, String>,
}

/// Semantic Knowledge Graph Index.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticGraphIndex {
    pub nodes: HashMap<String, GraphNode>,
    pub adj: HashMap<String, HashSet<String>>,
    pub rev_adj: HashMap<String, HashSet<String>>,
}

impl SemanticGraphIndex {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            adj: HashMap::new(),
            rev_adj: HashMap::new(),
        }
    }

    /// Adds a node to the index.
    pub fn add_node(&mut self, id: impl Into<String>, kind: NodeKind) {
        let id_str = id.into();
        self.nodes
            .entry(id_str.clone())
            .or_insert_with(|| GraphNode {
                id: id_str.clone(),
                kind,
                metadata: HashMap::new(),
            });
        self.adj.entry(id_str.clone()).or_default();
        self.rev_adj.entry(id_str).or_default();
    }

    /// Adds a directed dependency edge: `from` depends on `to`.
    pub fn add_edge(&mut self, from: &str, to: &str) {
        self.adj
            .entry(from.to_string())
            .or_default()
            .insert(to.to_string());
        self.rev_adj
            .entry(to.to_string())
            .or_default()
            .insert(from.to_string());
    }

    /// Returns transitive dependencies of a node (all nodes reachable following forward edges).
    pub fn transitive_dependencies(&self, start: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.to_string());

        while let Some(curr) = queue.pop_front() {
            if let Some(neighbors) = self.adj.get(&curr) {
                for n in neighbors {
                    if visited.insert(n.clone()) {
                        queue.push_back(n.clone());
                    }
                }
            }
        }
        visited
    }

    /// Checks if the dependency graph contains any cycles.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node in self.nodes.keys() {
            if !visited.contains(node) && self.cycle_dfs(node, &mut visited, &mut rec_stack) {
                return true;
            }
        }
        false
    }

    fn cycle_dfs(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(neighbors) = self.adj.get(node) {
            for n in neighbors {
                if !visited.contains(n) {
                    if self.cycle_dfs(n, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(n) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }
}
