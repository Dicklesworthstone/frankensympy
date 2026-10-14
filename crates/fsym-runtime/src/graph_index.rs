//! Semantic knowledge graph indexing and FrankenGraphDB projection adapter (WS16 / `docs/GRAPH_REASONING_SUBSTRATE.md`).
//!
//! Constitutional & Graph Substrate invariants:
//! - Derived projections (`frankengraphdb_v1`) are optional and NEVER authoritative for identity or proof.
//! - Every authoritative projection read binds to an `ingestion_watermark`, `source_manifest_root`, and `universe_digest`.
//! - Missing paths without an explicit completeness basis return `ReachabilityQueryResult::NotObserved`, never `DoesNotExist`.
//! - No graph vertex handle, insertion slot, or row ID ever enters `TermId`.
//! - Destruction of on-disk projection indexes is permitted ONLY for isolated test indexes bearing the explicit sentinel marker.

#![forbid(unsafe_code)]

use crate::workspace::SemanticWorkspace;
use fsym_assumptions::ImmutableAssumptionsSnapshot;
use fsym_core::{Expr, TermDag};
use fsym_proof_kernel::{DerivationTree, ProofRule, StepId, verify_derivation_independent};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const PROJECTION_ADAPTER_ID: &str = "frankengraphdb_v1";
pub const PROJECTION_SCHEMA_VERSION: u32 = 1;
pub const ISOLATED_TEST_INDEX_SENTINEL: &str = ".fsym_isolated_graph_projection_marker";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphIndexError {
    #[error("Cycle detected in semantic dependency graph: {0}")]
    CycleDetected(String),
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    #[error("Stale projection schema version: found {found}, expected {expected}")]
    StaleProjectionVersion { found: u32, expected: u32 },
    #[error(
        "Stale ingestion watermark: projection at {found}, authoritative manifest at {expected}"
    )]
    StaleWatermark { found: u64, expected: u64 },
    #[error("Source manifest root mismatch between projection and authoritative artifacts")]
    ManifestRootMismatch { found: [u8; 32], expected: [u8; 32] },
    #[error("Assumption universe context mismatch")]
    UniverseContextMismatch { found: [u8; 32], expected: [u8; 32] },
    #[error("Malicious or invalid graph index answer rejected: {0}")]
    MaliciousIndexAnswer(String),
    #[error("Authoritative derivation failed independent re-verification: {0}")]
    DerivationReverificationFailed(String),
    #[error("TermId invariant violation: graph indexing altered or entered semantic TermId")]
    TermIdContamination,
    #[error("Private object export denied for node {0}")]
    PrivateObjectExportDenied(String),
    #[error("Refused to destroy graph index outside isolated test fixture policy: {0}")]
    DestroyPolicyRefused(String),
    #[error("Graph projection adapter is unavailable")]
    GraphUnavailable,
    #[error("Graph projection I/O or serialization failure: {0}")]
    StorageFailure(String),
}

/// Node kind in the semantic knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    Workspace,
    Symbol,
    Theorem,
    Derivation,
    Term,
}

/// Typed edge kind from `docs/GRAPH_REASONING_SUBSTRATE.md` §3.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum GraphEdgeKind {
    OrderedChild(u32),
    DependsOn,
    Proves,
    UsesRule,
    UsesAssumption,
    Implies,
    Contradicts,
    Refines,
    CoercesTo,
    Invalidates,
    GeneratedFrom,
    AlternativeDerivation,
    MustPrecede,
    MayOverlap,
    BranchParent,
    ArtifactContains,
}

/// A typed directed edge in the semantic knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedGraphEdge {
    pub from: String,
    pub to: String,
    pub kind: GraphEdgeKind,
}

/// A node in the semantic knowledge graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub kind: NodeKind,
    pub metadata: HashMap<String, String>,
}

/// Portable certificate for a directed reachability path (`reachability_path_v1`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReachabilityPathCertificate {
    pub certificate_family: String,
    pub source: String,
    pub target: String,
    pub vertices: Vec<String>,
    pub edge_kinds: Vec<GraphEdgeKind>,
    pub snapshot_root: [u8; 32],
}

/// Portable certificate for a DAG topological ordering (`topological_order_v1`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologicalOrderCertificate {
    pub certificate_family: String,
    pub order: Vec<String>,
    pub snapshot_root: [u8; 32],
}

/// Portable certificate witnessing a directed cycle (`cycle_witness_v1`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CycleWitnessCertificate {
    pub certificate_family: String,
    pub cycle_vertices: Vec<String>,
}

/// Result of a reachability query against the graph substrate or projection adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReachabilityQueryResult {
    /// Target is reachable via an independently verified directed path.
    Reachable {
        path: Vec<String>,
        certificate: ReachabilityPathCertificate,
    },
    /// Target is provably unreachable under an explicit verified completeness basis.
    DoesNotExist {
        completeness_basis: String,
        snapshot_root: [u8; 32],
    },
    /// No path was observed, or completeness / projection availability is not established.
    /// Never carries negative-proof authority (`docs/GRAPH_REASONING_SUBSTRATE.md` §8).
    NotObserved { reason: String },
}

/// Semantic Knowledge Graph Index.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticGraphIndex {
    pub nodes: HashMap<String, GraphNode>,
    pub adj: HashMap<String, HashSet<String>>,
    pub rev_adj: HashMap<String, HashSet<String>>,
    #[serde(default)]
    pub typed_edges: BTreeSet<TypedGraphEdge>,
}

impl SemanticGraphIndex {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            adj: HashMap::new(),
            rev_adj: HashMap::new(),
            typed_edges: BTreeSet::new(),
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
        self.add_typed_edge(from, to, GraphEdgeKind::DependsOn);
    }

    /// Adds a typed directed edge between `from` and `to`.
    pub fn add_typed_edge(&mut self, from: &str, to: &str, kind: GraphEdgeKind) {
        self.adj
            .entry(from.to_string())
            .or_default()
            .insert(to.to_string());
        self.rev_adj
            .entry(to.to_string())
            .or_default()
            .insert(from.to_string());
        self.typed_edges.insert(TypedGraphEdge {
            from: from.to_string(),
            to: to.to_string(),
            kind,
        });
    }

    /// Deterministic snapshot root committing to canonical nodes and typed edges.
    pub fn snapshot_root(&self) -> [u8; 32] {
        let mut sorted_nodes: Vec<(&String, &NodeKind)> =
            self.nodes.iter().map(|(k, v)| (k, &v.kind)).collect();
        sorted_nodes.sort_by(|a, b| a.0.cmp(b.0));

        let mut sorted_adj: Vec<(&String, Vec<&String>)> = self
            .adj
            .iter()
            .map(|(k, set)| {
                let mut targets: Vec<&String> = set.iter().collect();
                targets.sort();
                (k, targets)
            })
            .collect();
        sorted_adj.sort_by(|a, b| a.0.cmp(b.0));

        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.graph.snapshot.v1:");
        hasher.update(&serde_json::to_vec(&sorted_nodes).unwrap_or_default());
        hasher.update(&serde_json::to_vec(&sorted_adj).unwrap_or_default());
        hasher.update(&serde_json::to_vec(&self.typed_edges).unwrap_or_default());
        *hasher.finalize().as_bytes()
    }

    /// Returns transitive dependencies of a node (all nodes reachable following forward edges).
    pub fn transitive_dependencies(&self, start: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.to_string());

        while let Some(curr) = queue.pop_front() {
            if let Some(neighbors) = self.adj.get(&curr) {
                let mut ordered: Vec<&String> = neighbors.iter().collect();
                ordered.sort();
                for n in ordered {
                    if visited.insert(n.clone()) {
                        queue.push_back(n.clone());
                    }
                }
            }
        }
        visited
    }

    /// Finds a canonical shortest directed path from `start` to `target` using `content_id_ascending_v1` tie-breaking.
    pub fn shortest_path(&self, start: &str, target: &str) -> Option<Vec<String>> {
        if !self.nodes.contains_key(start) || !self.nodes.contains_key(target) {
            return None;
        }
        if start == target {
            return Some(vec![start.to_string()]);
        }
        let mut visited = HashSet::new();
        let mut parent: HashMap<String, String> = HashMap::new();
        let mut queue = VecDeque::new();
        visited.insert(start.to_string());
        queue.push_back(start.to_string());

        while let Some(curr) = queue.pop_front() {
            if let Some(neighbors) = self.adj.get(&curr) {
                let mut ordered: Vec<&String> = neighbors.iter().collect();
                ordered.sort();
                for next in ordered {
                    if visited.insert(next.clone()) {
                        parent.insert(next.clone(), curr.clone());
                        if next == target {
                            let mut path = vec![target.to_string()];
                            let mut cur = target.to_string();
                            while let Some(p) = parent.get(&cur) {
                                path.push(p.clone());
                                if p == start {
                                    break;
                                }
                                cur = p.clone();
                            }
                            path.reverse();
                            return Some(path);
                        }
                        queue.push_back(next.clone());
                    }
                }
            }
        }
        None
    }

    /// Computes a deterministic topological order (`content_id_ascending_v1`), or errors if cyclic.
    pub fn topological_order(&self) -> Result<TopologicalOrderCertificate, GraphIndexError> {
        if self.has_cycle() {
            return Err(GraphIndexError::CycleDetected(
                "cannot produce topological order on cyclic graph".into(),
            ));
        }
        let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
        for node_id in self.nodes.keys() {
            in_degree.insert(node_id.clone(), 0);
        }
        for (u, targets) in &self.adj {
            if !self.nodes.contains_key(u) {
                continue;
            }
            for v in targets {
                if let Some(deg) = in_degree.get_mut(v) {
                    *deg += 1;
                }
            }
        }
        let mut ready: BTreeSet<String> = in_degree
            .iter()
            .filter_map(|(k, &deg)| if deg == 0 { Some(k.clone()) } else { None })
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(curr) = ready.pop_first() {
            order.push(curr.clone());
            if let Some(neighbors) = self.adj.get(&curr) {
                let mut sorted_n: Vec<&String> = neighbors.iter().collect();
                sorted_n.sort();
                for n in sorted_n {
                    if let Some(deg) = in_degree.get_mut(n) {
                        *deg -= 1;
                        if *deg == 0 {
                            ready.insert(n.clone());
                        }
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(GraphIndexError::CycleDetected(
                "remaining nodes have non-zero in-degree".into(),
            ));
        }
        Ok(TopologicalOrderCertificate {
            certificate_family: "topological_order_v1".into(),
            order,
            snapshot_root: self.snapshot_root(),
        })
    }

    /// Checks if the dependency graph contains any cycles.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        let mut keys: Vec<&String> = self.nodes.keys().collect();
        keys.sort();
        for node in keys {
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
            let mut ordered: Vec<&String> = neighbors.iter().collect();
            ordered.sort();
            for n in ordered {
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

/// Independent runtime-free verifier for `reachability_path_v1` certificates.
pub fn verify_reachability_path(
    graph: &SemanticGraphIndex,
    cert: &ReachabilityPathCertificate,
) -> Result<(), GraphIndexError> {
    if cert.certificate_family != "reachability_path_v1" {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "unexpected reachability certificate family".into(),
        ));
    }
    if cert.snapshot_root != graph.snapshot_root() {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "reachability certificate snapshot_root mismatch".into(),
        ));
    }
    if cert.vertices.is_empty() {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "reachability path cannot be empty".into(),
        ));
    }
    if cert.vertices.first() != Some(&cert.source) || cert.vertices.last() != Some(&cert.target) {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "reachability path endpoints do not match declared source/target".into(),
        ));
    }
    for v in &cert.vertices {
        if !graph.nodes.contains_key(v) {
            return Err(GraphIndexError::MaliciousIndexAnswer(format!(
                "reachability path references unknown vertex {v}"
            )));
        }
    }
    if cert.vertices.len() == 1 {
        if !cert.edge_kinds.is_empty() {
            return Err(GraphIndexError::MaliciousIndexAnswer(
                "single-vertex path cannot carry edges".into(),
            ));
        }
        return Ok(());
    }
    if cert.edge_kinds.len() + 1 != cert.vertices.len() {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "edge_kinds length must equal vertices.len() - 1".into(),
        ));
    }
    for i in 0..(cert.vertices.len() - 1) {
        let u = &cert.vertices[i];
        let v = &cert.vertices[i + 1];
        let has_adj = graph.adj.get(u).is_some_and(|set| set.contains(v));
        if !has_adj {
            return Err(GraphIndexError::MaliciousIndexAnswer(format!(
                "missing directed edge {u} -> {v} in authoritative graph"
            )));
        }
    }
    Ok(())
}

/// Independent runtime-free verifier for `topological_order_v1` certificates.
pub fn verify_topological_order(
    graph: &SemanticGraphIndex,
    cert: &TopologicalOrderCertificate,
) -> Result<(), GraphIndexError> {
    if cert.certificate_family != "topological_order_v1" {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "unexpected topological order certificate family".into(),
        ));
    }
    if cert.snapshot_root != graph.snapshot_root() {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "topological certificate snapshot_root mismatch".into(),
        ));
    }
    if cert.order.len() != graph.nodes.len() {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "topological order length does not match graph node count".into(),
        ));
    }
    let mut positions: HashMap<&str, usize> = HashMap::with_capacity(cert.order.len());
    for (idx, node) in cert.order.iter().enumerate() {
        if !graph.nodes.contains_key(node) {
            return Err(GraphIndexError::MaliciousIndexAnswer(format!(
                "topological order contains unknown node {node}"
            )));
        }
        if positions.insert(node.as_str(), idx).is_some() {
            return Err(GraphIndexError::MaliciousIndexAnswer(format!(
                "duplicate node {node} in topological order"
            )));
        }
    }
    for (u, neighbors) in &graph.adj {
        let Some(&pos_u) = positions.get(u.as_str()) else {
            continue;
        };
        for v in neighbors {
            let Some(&pos_v) = positions.get(v.as_str()) else {
                return Err(GraphIndexError::MaliciousIndexAnswer(format!(
                    "edge endpoint {v} missing from topological order"
                )));
            };
            if pos_u >= pos_v {
                return Err(GraphIndexError::MaliciousIndexAnswer(format!(
                    "edge {u} -> {v} violates topological order ({pos_u} >= {pos_v})"
                )));
            }
        }
    }
    Ok(())
}

/// Independent runtime-free verifier for `cycle_witness_v1` certificates.
pub fn verify_cycle_witness(
    graph: &SemanticGraphIndex,
    cert: &CycleWitnessCertificate,
) -> Result<(), GraphIndexError> {
    if cert.certificate_family != "cycle_witness_v1" || cert.cycle_vertices.len() < 2 {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "invalid cycle witness header or length".into(),
        ));
    }
    if cert.cycle_vertices.first() != cert.cycle_vertices.last() {
        return Err(GraphIndexError::MaliciousIndexAnswer(
            "cycle witness is not closed".into(),
        ));
    }
    for i in 0..(cert.cycle_vertices.len() - 1) {
        let u = &cert.cycle_vertices[i];
        let v = &cert.cycle_vertices[i + 1];
        if !graph.adj.get(u).is_some_and(|set| set.contains(v)) {
            return Err(GraphIndexError::MaliciousIndexAnswer(format!(
                "cycle witness edge {u} -> {v} does not exist"
            )));
        }
    }
    Ok(())
}

/// Authoritative workspace manifest from which a `FrankenGraphDbProjectionAdapter` is built or rebuilt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthoritativeWorkspaceManifest {
    pub workspace_id: String,
    pub ingestion_watermark: u64,
    pub universe_digest: [u8; 32],
    pub assumptions: ImmutableAssumptionsSnapshot,
    pub completeness_basis: Option<String>,
    pub bindings: BTreeMap<String, Expr>,
    pub derivations: BTreeMap<String, DerivationTree>,
    pub private_objects: BTreeSet<String>,
    pub source_manifest_root: [u8; 32],
}

fn expr_term_digest(expr: &Expr) -> [u8; 32] {
    let mut dag = TermDag::new();
    let mut hasher = blake3::Hasher::new();
    match dag.insert_expr(expr) {
        Ok(term_id) => {
            hasher.update(b"fsym.expr.term_id.v1:");
            hasher.update(&term_id.to_binary());
        }
        Err(_) => {
            hasher.update(b"fsym.expr.fallback.digest.v1:");
            if let Ok(bytes) = serde_json::to_vec(expr) {
                hasher.update(&bytes);
            }
        }
    }
    *hasher.finalize().as_bytes()
}

fn rule_premise_steps(rule: &ProofRule) -> Vec<StepId> {
    match rule {
        ProofRule::Reflexivity(_)
        | ProofRule::ContextPredicate { .. }
        | ProofRule::ContextDomain { .. }
        | ProofRule::DefinitionalReduction { .. }
        | ProofRule::CertificateLemma { .. } => Vec::new(),
        ProofRule::Symmetry(s) => vec![*s],
        ProofRule::Transitivity(a, b) => vec![*a, *b],
        ProofRule::CongruenceAdd(args)
        | ProofRule::CongruenceMul(args)
        | ProofRule::CongruenceFunction { args, .. } => args.clone(),
        ProofRule::CongruencePow { base, exp } => vec![*base, *exp],
        ProofRule::Substitution { step, .. } => vec![*step],
    }
}

impl AuthoritativeWorkspaceManifest {
    /// Extracts a deterministic authoritative manifest from a `SemanticWorkspace`.
    pub fn from_workspace(
        workspace: &SemanticWorkspace,
        ingestion_watermark: u64,
        completeness_basis: Option<String>,
        private_objects: BTreeSet<String>,
    ) -> Self {
        let mut bindings = BTreeMap::new();
        for (sym, expr) in &workspace.bindings {
            bindings.insert(sym.name.clone(), expr.clone());
        }
        let mut derivations = BTreeMap::new();
        for (idx, deriv) in workspace.derivations.iter().enumerate() {
            let d_hex: String = deriv
                .digest()
                .iter()
                .take(8)
                .map(|b| format!("{b:02x}"))
                .collect();
            derivations.insert(format!("deriv_{idx}_{d_hex}"), deriv.clone());
        }
        let universe_digest = workspace.assumptions.digest();
        let mut manifest = Self {
            workspace_id: format!("ws:{}", workspace.branch_name),
            ingestion_watermark,
            universe_digest,
            assumptions: (*workspace.assumptions).clone(),
            completeness_basis,
            bindings,
            derivations,
            private_objects,
            source_manifest_root: [0u8; 32],
        };
        manifest.source_manifest_root = manifest.compute_root();
        manifest
    }

    pub fn compute_root(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.authoritative.workspace.manifest.v1:");
        hasher.update(self.workspace_id.as_bytes());
        hasher.update(&self.ingestion_watermark.to_le_bytes());
        hasher.update(&self.universe_digest);
        if let Some(basis) = &self.completeness_basis {
            hasher.update(b"complete:");
            hasher.update(basis.as_bytes());
        } else {
            hasher.update(b"incomplete");
        }
        for (sym, expr) in &self.bindings {
            hasher.update(sym.as_bytes());
            hasher.update(&expr_term_digest(expr));
        }
        for (did, deriv) in &self.derivations {
            hasher.update(did.as_bytes());
            hasher.update(&deriv.digest());
        }
        for priv_obj in &self.private_objects {
            hasher.update(priv_obj.as_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

/// Explicit test-only policy required before destroying an on-disk isolated projection index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestFixtureDestroyPolicy {
    AllowIsolatedTestIndex { dir: PathBuf },
}

/// Persisted state of the `frankengraphdb_v1` derived projection adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrankenGraphDbProjection {
    pub adapter_id: String,
    pub schema_version: u32,
    pub source_manifest_root: [u8; 32],
    pub ingestion_watermark: u64,
    pub universe_digest: [u8; 32],
    pub completeness_basis: Option<String>,
    pub index: SemanticGraphIndex,
    pub verified_derivation_digests: BTreeMap<String, [u8; 32]>,
}

/// Derived projection adapter (`frankengraphdb_v1`) backed by an optional isolated store directory.
#[derive(Debug, Clone)]
pub struct FrankenGraphDbProjectionAdapter {
    storage_dir: Option<PathBuf>,
    available: bool,
    projection: Option<FrankenGraphDbProjection>,
}

impl FrankenGraphDbProjectionAdapter {
    /// Creates an in-memory projection adapter without on-disk persistence.
    pub fn new_in_memory() -> Self {
        Self {
            storage_dir: None,
            available: true,
            projection: None,
        }
    }

    /// Creates an isolated on-disk projection adapter for test fixtures, writing the sentinel marker.
    pub fn create_isolated_test_adapter(dir: impl Into<PathBuf>) -> Result<Self, GraphIndexError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| GraphIndexError::StorageFailure(e.to_string()))?;
        let sentinel = dir.join(ISOLATED_TEST_INDEX_SENTINEL);
        std::fs::write(&sentinel, b"fsym.isolated.graph.projection.v1\n")
            .map_err(|e| GraphIndexError::StorageFailure(e.to_string()))?;
        Ok(Self {
            storage_dir: Some(dir),
            available: true,
            projection: None,
        })
    }

    /// Simulates or toggles projection adapter availability (for testing graceful degradation).
    pub fn set_available(&mut self, available: bool) {
        self.available = available;
    }

    pub fn is_available(&self) -> bool {
        self.available
    }

    pub fn projection(&self) -> Option<&FrankenGraphDbProjection> {
        self.projection.as_ref()
    }

    /// Destroys an isolated test index on disk ONLY if `policy` matches `self.storage_dir`
    /// and the directory contains `ISOLATED_TEST_INDEX_SENTINEL`.
    pub fn destroy_isolated_test_index(
        &mut self,
        policy: &TestFixtureDestroyPolicy,
    ) -> Result<(), GraphIndexError> {
        let TestFixtureDestroyPolicy::AllowIsolatedTestIndex { dir: allowed_dir } = policy;
        let Some(current_dir) = &self.storage_dir else {
            return Err(GraphIndexError::DestroyPolicyRefused(
                "adapter has no on-disk test directory".into(),
            ));
        };
        if current_dir != allowed_dir {
            return Err(GraphIndexError::DestroyPolicyRefused(format!(
                "policy path {} does not match adapter path {}",
                allowed_dir.display(),
                current_dir.display()
            )));
        }
        let sentinel = current_dir.join(ISOLATED_TEST_INDEX_SENTINEL);
        if !sentinel.is_file() {
            return Err(GraphIndexError::DestroyPolicyRefused(format!(
                "missing sentinel {} in {}",
                ISOLATED_TEST_INDEX_SENTINEL,
                current_dir.display()
            )));
        }
        let snapshot_file = current_dir.join("projection_snapshot.json");
        if snapshot_file.is_file() {
            // Overwrite with empty marker rather than rm -rf (`AGENTS.md` §4 safe cleanup).
            std::fs::write(&snapshot_file, b"")
                .map_err(|e| GraphIndexError::StorageFailure(e.to_string()))?;
        }
        self.projection = None;
        Ok(())
    }

    /// Rebuilds the entire `frankengraphdb_v1` projection from authoritative workspace artifacts,
    /// re-verifying every accepted derivation from canonical objects and proving `TermId` invariance.
    pub fn rebuild_from_authoritative(
        &mut self,
        manifest: &AuthoritativeWorkspaceManifest,
    ) -> Result<&FrankenGraphDbProjection, GraphIndexError> {
        if !self.available {
            return Err(GraphIndexError::GraphUnavailable);
        }
        if manifest.compute_root() != manifest.source_manifest_root {
            return Err(GraphIndexError::ManifestRootMismatch {
                found: manifest.source_manifest_root,
                expected: manifest.compute_root(),
            });
        }

        let mut index = SemanticGraphIndex::new();
        index.add_node(manifest.workspace_id.clone(), NodeKind::Workspace);

        // 1. Index bindings and verify TermId invariance before/after graph insertion.
        for (sym_name, expr) in &manifest.bindings {
            let term_id_before = expr_term_digest(expr);
            let sym_node = format!("sym:{sym_name}");
            let hex_id: String = term_id_before
                .iter()
                .take(8)
                .map(|b| format!("{b:02x}"))
                .collect();
            let term_node = format!("term:{hex_id}");

            index.add_node(sym_node.clone(), NodeKind::Symbol);
            index.add_node(term_node.clone(), NodeKind::Term);
            index.add_typed_edge(
                &manifest.workspace_id,
                &sym_node,
                GraphEdgeKind::ArtifactContains,
            );
            index.add_typed_edge(&sym_node, &term_node, GraphEdgeKind::DependsOn);

            for free_sym in expr.free_symbols() {
                let dep_sym_node = format!("sym:{}", free_sym.name);
                index.add_node(dep_sym_node.clone(), NodeKind::Symbol);
                index.add_typed_edge(&term_node, &dep_sym_node, GraphEdgeKind::DependsOn);
            }

            let term_id_after = expr_term_digest(expr);
            if term_id_before != term_id_after {
                return Err(GraphIndexError::TermIdContamination);
            }
        }

        // 2. Re-verify every accepted derivation from canonical objects and index its proof DAG.
        let mut verified_derivation_digests = BTreeMap::new();
        for (deriv_id, deriv) in &manifest.derivations {
            let verified_claim = verify_derivation_independent(deriv, &manifest.assumptions)
                .map_err(|e| GraphIndexError::DerivationReverificationFailed(e.to_string()))?;

            let deriv_node = format!("deriv:{deriv_id}");
            let thm_node = format!("thm:{deriv_id}");
            index.add_node(deriv_node.clone(), NodeKind::Derivation);
            index.add_node(thm_node.clone(), NodeKind::Theorem);
            index.add_typed_edge(
                &manifest.workspace_id,
                &thm_node,
                GraphEdgeKind::ArtifactContains,
            );
            index.add_typed_edge(&thm_node, &deriv_node, GraphEdgeKind::Proves);

            for (pos, step) in deriv.steps.iter().enumerate() {
                let step_node = format!("deriv:{deriv_id}:step:{}", step.id.0);
                index.add_node(step_node.clone(), NodeKind::Derivation);
                index.add_typed_edge(
                    &deriv_node,
                    &step_node,
                    GraphEdgeKind::OrderedChild(pos as u32),
                );
                for prem in rule_premise_steps(&step.rule) {
                    let prem_node = format!("deriv:{deriv_id}:step:{}", prem.0);
                    index.add_node(prem_node.clone(), NodeKind::Derivation);
                    index.add_typed_edge(&step_node, &prem_node, GraphEdgeKind::DependsOn);
                }
            }

            // Link theorem to free symbols in its verified claim
            let claim_syms = match &verified_claim {
                fsym_proof_kernel::Claim::Equality { lhs, rhs }
                | fsym_proof_kernel::Claim::AlgebraicIdentity { lhs, rhs } => {
                    let mut s = lhs.free_symbols();
                    s.extend(rhs.free_symbols());
                    s
                }
                fsym_proof_kernel::Claim::PredicateHold { expr, .. }
                | fsym_proof_kernel::Claim::DomainMembership { expr, .. }
                | fsym_proof_kernel::Claim::NonZero(expr) => expr.free_symbols(),
            };
            for sym in claim_syms {
                let sym_node = format!("sym:{}", sym.name);
                index.add_node(sym_node.clone(), NodeKind::Symbol);
                index.add_typed_edge(&deriv_node, &sym_node, GraphEdgeKind::DependsOn);
            }

            verified_derivation_digests.insert(deriv_id.clone(), deriv.digest());
        }

        let proj = FrankenGraphDbProjection {
            adapter_id: PROJECTION_ADAPTER_ID.into(),
            schema_version: PROJECTION_SCHEMA_VERSION,
            source_manifest_root: manifest.source_manifest_root,
            ingestion_watermark: manifest.ingestion_watermark,
            universe_digest: manifest.universe_digest,
            completeness_basis: manifest.completeness_basis.clone(),
            index,
            verified_derivation_digests,
        };

        if let Some(dir) = &self.storage_dir {
            let path = dir.join("projection_snapshot.json");
            let bytes = serde_json::to_vec_pretty(&proj)
                .map_err(|e| GraphIndexError::StorageFailure(e.to_string()))?;
            std::fs::write(&path, bytes)
                .map_err(|e| GraphIndexError::StorageFailure(e.to_string()))?;
        }

        self.projection = Some(proj);
        Ok(self.projection.as_ref().expect("projection just stored"))
    }

    /// Loads the persisted projection from the isolated directory and validates its header.
    pub fn load_from_disk(
        dir: impl AsRef<Path>,
        expected_watermark: u64,
        expected_manifest_root: [u8; 32],
        expected_universe_digest: [u8; 32],
    ) -> Result<Self, GraphIndexError> {
        let dir_path = dir.as_ref().to_path_buf();
        let path = dir_path.join("projection_snapshot.json");
        let bytes =
            std::fs::read(&path).map_err(|e| GraphIndexError::StorageFailure(e.to_string()))?;
        if bytes.is_empty() {
            return Err(GraphIndexError::GraphUnavailable);
        }
        let proj: FrankenGraphDbProjection = serde_json::from_slice(&bytes)
            .map_err(|e| GraphIndexError::StorageFailure(e.to_string()))?;
        validate_projection_freshness(
            &proj,
            expected_watermark,
            expected_manifest_root,
            expected_universe_digest,
        )?;
        Ok(Self {
            storage_dir: Some(dir_path),
            available: true,
            projection: Some(proj),
        })
    }

    /// Queries reachability between `from` and `to` with freshness, context, and completeness checks.
    pub fn query_reachability(
        &self,
        from: &str,
        to: &str,
        expected_watermark: u64,
        expected_manifest_root: [u8; 32],
        expected_universe_digest: [u8; 32],
    ) -> Result<ReachabilityQueryResult, GraphIndexError> {
        if !self.available {
            return Ok(ReachabilityQueryResult::NotObserved {
                reason: "graph_projection_unavailable".into(),
            });
        }
        let Some(proj) = &self.projection else {
            return Ok(ReachabilityQueryResult::NotObserved {
                reason: "graph_projection_not_built".into(),
            });
        };
        validate_projection_freshness(
            proj,
            expected_watermark,
            expected_manifest_root,
            expected_universe_digest,
        )?;

        if let Some(path) = proj.index.shortest_path(from, to) {
            let edge_kinds = vec![GraphEdgeKind::DependsOn; path.len().saturating_sub(1)];
            let cert = ReachabilityPathCertificate {
                certificate_family: "reachability_path_v1".into(),
                source: from.to_string(),
                target: to.to_string(),
                vertices: path.clone(),
                edge_kinds,
                snapshot_root: proj.index.snapshot_root(),
            };
            verify_reachability_path(&proj.index, &cert)?;
            return Ok(ReachabilityQueryResult::Reachable {
                path,
                certificate: cert,
            });
        }

        if let Some(basis) = &proj.completeness_basis
            && proj.index.nodes.contains_key(from)
            && proj.index.nodes.contains_key(to)
        {
            return Ok(ReachabilityQueryResult::DoesNotExist {
                completeness_basis: basis.clone(),
                snapshot_root: proj.index.snapshot_root(),
            });
        }
        Ok(ReachabilityQueryResult::NotObserved {
            reason: "no_path_observed_without_complete_basis".into(),
        })
    }

    /// Verifies an untrusted/candidate path returned by an external index against the authoritative projection.
    pub fn verify_candidate_index_answer(
        &self,
        source: &str,
        target: &str,
        candidate_vertices: Vec<String>,
    ) -> Result<ReachabilityPathCertificate, GraphIndexError> {
        let Some(proj) = &self.projection else {
            return Err(GraphIndexError::GraphUnavailable);
        };
        let edge_kinds = vec![GraphEdgeKind::DependsOn; candidate_vertices.len().saturating_sub(1)];
        let cert = ReachabilityPathCertificate {
            certificate_family: "reachability_path_v1".into(),
            source: source.to_string(),
            target: target.to_string(),
            vertices: candidate_vertices,
            edge_kinds,
            snapshot_root: proj.index.snapshot_root(),
        };
        verify_reachability_path(&proj.index, &cert)?;
        Ok(cert)
    }

    /// Exports a public dependency slice starting from `roots`, refusing if any root or reachable
    /// dependency is listed in `manifest.private_objects`.
    pub fn export_public_slice(
        &self,
        manifest: &AuthoritativeWorkspaceManifest,
        roots: &[String],
    ) -> Result<BTreeSet<String>, GraphIndexError> {
        let Some(proj) = &self.projection else {
            return Err(GraphIndexError::GraphUnavailable);
        };
        validate_projection_freshness(
            proj,
            manifest.ingestion_watermark,
            manifest.source_manifest_root,
            manifest.universe_digest,
        )?;
        let mut slice = BTreeSet::new();
        for root in roots {
            if manifest.private_objects.contains(root) {
                return Err(GraphIndexError::PrivateObjectExportDenied(root.clone()));
            }
            slice.insert(root.clone());
            for dep in proj.index.transitive_dependencies(root) {
                if manifest.private_objects.contains(&dep) {
                    return Err(GraphIndexError::PrivateObjectExportDenied(dep));
                }
                slice.insert(dep);
            }
        }
        Ok(slice)
    }
}

fn validate_projection_freshness(
    proj: &FrankenGraphDbProjection,
    expected_watermark: u64,
    expected_manifest_root: [u8; 32],
    expected_universe_digest: [u8; 32],
) -> Result<(), GraphIndexError> {
    if proj.schema_version != PROJECTION_SCHEMA_VERSION {
        return Err(GraphIndexError::StaleProjectionVersion {
            found: proj.schema_version,
            expected: PROJECTION_SCHEMA_VERSION,
        });
    }
    if proj.universe_digest != expected_universe_digest {
        return Err(GraphIndexError::UniverseContextMismatch {
            found: proj.universe_digest,
            expected: expected_universe_digest,
        });
    }
    if proj.ingestion_watermark < expected_watermark {
        return Err(GraphIndexError::StaleWatermark {
            found: proj.ingestion_watermark,
            expected: expected_watermark,
        });
    }
    if proj.source_manifest_root != expected_manifest_root {
        return Err(GraphIndexError::ManifestRootMismatch {
            found: proj.source_manifest_root,
            expected: expected_manifest_root,
        });
    }
    Ok(())
}
