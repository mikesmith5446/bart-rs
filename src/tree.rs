//! A Binary Decision Tree is the core data structure for Bayesian Additive
//! Regression Trees (BART). The tree is implemented using an array (vector)
//! representation.

use core::fmt;
//use std::cmp::Ordering;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// A `DecisionTree` is an array-based implementation of the binary decision tree.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionTree {
    pub feature: Vec<usize>,
    pub threshold: Vec<f64>,
    pub value: Vec<f64>,
    pub left_child: Vec<i32>,
    pub right_child: Vec<i32>,
    pub parent: Vec<i32>,
    pub n_left: Vec<i32>,   // optional, if you want excluded weighting
    pub n_right: Vec<i32>,  // optional
    pub leaf_id: Vec<i32>,   // leaf nodes: >= 0, internal nodes: -1
    pub next_leaf_id: i32,   // monotonically increasing
}

/// Represents errors related to binary decision tree operations.
#[derive(Debug)]
pub enum TreeError {
    /// When attempting to split a leaf node, if the node is not a leaf.
    NonLeafSplit,
    /// When attempting to split a leaf node, if the index is valid or not
    InvalidNodeIndex,
}

impl fmt::Display for TreeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TreeError::NonLeafSplit => write!(f, "Cannot split a non-leaf node"),
            TreeError::InvalidNodeIndex => write!(f, "Node index does not exist"),
        }
    }
}

pub struct TreeDumpParts {
    pub split_feature: Vec<i32>,
    pub split_value: Vec<f64>,
    pub left_child: Vec<i32>,
    pub right_child: Vec<i32>,
    pub leaf_value: Vec<f64>,
    pub n_left: Vec<i32>,
    pub n_right: Vec<i32>,
    pub root_index: i32,
}


impl DecisionTree {
    /// Creates a new `DecisionTree` with an initial value set as the root node.
    /// A decision tree is implemented as three parallel vectors.
    ///
    /// Using parallel vectors allows for a more cache-efficient implementation.
    /// Moreover, index accessing allows one to more easily avoid borrow checker
    /// issues related to classical recursive binary tree implementations.
    ///
    /// The `i-th` element of each vector holds information about node `i`. Node 0
    /// is the tree's root. Some of the vectors only apply to either leaves or
    /// split nodes. In this case, the values of the nodes of the other vectors is
    /// arbitrary. For example, `feature` and `threshold` vectors only apply to
    /// split nodes. The values for leaf nodes in these vectors are therefore
    /// arbitrary.
    pub fn new(init_value: f64) -> Self {
        Self {
            feature: vec![0],
            threshold: vec![0.0],
            value: vec![init_value],
            left_child: vec![-1],
            right_child: vec![-1],
            n_left: vec![0],
            n_right: vec![0],
            parent: vec![-1],
            leaf_id: vec![0],
            next_leaf_id: 1,
        }
    }

    /// Adds a new node to the `DecisionTree` and returns the location of _this_ node in the
    /// `feature` vector.
    pub fn add_node(&mut self, feature: usize, threshold: f64, value: f64) -> usize {
        let node_id = self.feature.len();
        self.feature.push(feature);
        self.threshold.push(threshold);
        self.value.push(value);
        self.left_child.push(-1);
        self.right_child.push(-1);
        self.n_left.push(0);
        self.n_right.push(0);
        self.parent.push(-1);
        self.leaf_id.push(self.next_leaf_id);
        self.next_leaf_id += 1;
        node_id
    }

    /// Computes the left child index of _this_ node.
    pub fn left_child(&self, index: usize) -> Option<usize> {
        let v = self.left_child[index];
        if v >= 0 { Some(v as usize) } else { None }
    }

    /// Computes the right child index of _this_ node.
    pub fn right_child(&self, index: usize) -> Option<usize> {
        let v = self.right_child[index];
        if v >= 0 { Some(v as usize) } else { None }
    }

    /// Checks whether the passed index is a leaf node.
    ///
    /// An index is a leaf node if both of its potential children are outside
    /// the valid array bounds.
    pub fn is_leaf(&self, index: usize) -> bool {
        self.left_child(index).is_none() && self.right_child(index).is_none()
    }

    /// Computes the depth of _this_ node in the `DecisionTree`.
    #[inline]
    pub fn node_depth(&self, index: usize) -> usize {
        let mut depth = 0usize;
        let mut cur = index as i32;
        while cur > 0 {
            depth += 1;
            cur = self.parent[cur as usize];
        }
        depth
    }

    /// Splits a leaf node into an internal node.
    pub fn split_node(
        &mut self,
        node_index: usize,
        feature: usize,
        threshold: f64,
        left_value: f64,
        right_value: f64,
        left_count: usize,
        right_count: usize,
    ) -> Result<(usize, usize), TreeError> {
        if node_index >= self.value.len() {
            return Err(TreeError::InvalidNodeIndex);
        }

        if !self.is_leaf(node_index) {
            return Err(TreeError::NonLeafSplit);
        }

        // Update the current node
        self.feature[node_index] = feature;
        self.threshold[node_index] = threshold;

        // Add new left and right leaf nodes
        let left_child_index = self.add_node(0, 0.0, left_value);
        let right_child_index = self.add_node(0, 0.0, right_value);

        // Debug-only sanity check: newly created nodes must be leaves
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.leaf_id[left_child_index] >= 0);
            debug_assert!(self.leaf_id[right_child_index] >= 0);
        }

        // set explicit pointers
        self.left_child[node_index] = left_child_index as i32;
        self.right_child[node_index] = right_child_index as i32;

        // optional counts for excluded weighting
        self.n_left[node_index] = left_count as i32;
        self.n_right[node_index] = right_count as i32;

        self.parent[left_child_index] = node_index as i32;
        self.parent[right_child_index] = node_index as i32;
        self.leaf_id[node_index] = -1;
        
        
        #[cfg(debug_assertions)]
        self.validate_invariants();
        Ok((left_child_index, right_child_index))
    }
    #[inline]
    pub fn is_split(&self, index: usize) -> bool {
        !self.is_leaf(index)
    }

    #[inline]
    pub fn children(&self, index: usize) -> Option<(usize, usize)> {
        match (self.left_child(index), self.right_child(index)) {
            (Some(l), Some(r)) => Some((l, r)),
            _ => None,
        }
    }

    #[inline]
    pub fn nvalue(&self, index: usize) -> i32 {
        self.n_left[index] + self.n_right[index]
    }

    #[inline]
    pub fn is_leaf_id(&self, index: usize) -> bool {
        self.leaf_id[index] >= 0
    }

    /// Predict the output given an input `sample`.
    #[inline]
    pub fn predict(&self, sample: &[f64]) -> f64 {
        let mut node: usize = 0;

        loop {
            if self.is_leaf(node) {
                return self.value[node];
            }

            let feature = self.feature[node];
            let threshold = self.threshold[node];

            // NaN-safe branching (avoids partial_cmp().unwrap() panic)
            let xi = sample[feature];
            if xi.is_nan() {
                // choose a consistent default direction for NaNs
                node = self.right_child(node).unwrap();
            } else if xi < threshold {
                node = self.left_child(node).unwrap();
            } else {
                node = self.right_child(node).unwrap();
            }
        }
    }

    /// Predict with excluded-variable weighting (PyMC parity).
    ///
    /// If the current node splits on a feature that is excluded, we traverse both branches and
    /// weight them by n_left/n_right proportions.
    ///
    /// `excluded_mask` must have length >= number of features and contain `true` for excluded features.
    pub fn predict_excluded_mask(&self, sample: &[f64], excluded_mask: &[bool]) -> f64 {
        let mut acc = 0.0;

        // stack holds (node_index, weight)
        let mut stack: Vec<(usize, f64)> = Vec::with_capacity(32);
        stack.push((0usize, 1.0));

        while let Some((node, w)) = stack.pop() {
            if self.is_leaf(node) {
                acc += w * self.value[node];
                continue;
            }

            let feat = self.feature[node];

            if feat < excluded_mask.len() && excluded_mask[feat] {
                // excluded => weighted mixture down both branches
                let n_l = self.n_left[node] as f64;
                let n_r = self.n_right[node] as f64;
                let tot = n_l + n_r;

                let (wl, wr) = if tot > 0.0 {
                    (w * (n_l / tot), w * (n_r / tot))
                } else {
                    // fallback if counts aren't set
                    (0.5 * w, 0.5 * w)
                };

                let (l, r) = self
                    .children(node)
                    .expect("internal node must have two children");
                stack.push((l, wl));
                stack.push((r, wr));
            } else {
                // normal deterministic traversal (NaN -> right)
                let thr = self.threshold[node];
                let xi = sample[feat];

                let next = if xi.is_nan() || xi >= thr {
                    self.right_child(node).unwrap()
                } else {
                    self.left_child(node).unwrap()
                };

                stack.push((next, w));
            }
        }

        acc
    }

    /// Helper for ArrayView row when it is not contiguous.
    fn predict_excluded_row_view(&self, row: &ArrayView1<'_, f64>, excluded_mask: &[bool]) -> f64 {
        let mut acc = 0.0;
        let mut stack: Vec<(usize, f64)> = Vec::with_capacity(32);
        stack.push((0usize, 1.0));

        while let Some((node, w)) = stack.pop() {
            if self.is_leaf(node) {
                acc += w * self.value[node];
                continue;
            }

            let feat = self.feature[node];

            if feat < excluded_mask.len() && excluded_mask[feat] {
                let n_l = self.n_left[node] as f64;
                let n_r = self.n_right[node] as f64;
                let tot = n_l + n_r;

                let (wl, wr) = if tot > 0.0 {
                    (w * (n_l / tot), w * (n_r / tot))
                } else {
                    (0.5 * w, 0.5 * w)
                };

                let (l, r) = self
                    .children(node)
                    .expect("internal node must have two children");
                stack.push((l, wl));
                stack.push((r, wr));
            } else {
                let thr = self.threshold[node];
                let xi = row[feat];

                let next = if xi.is_nan() || xi >= thr {
                    self.right_child(node).unwrap()
                } else {
                    self.left_child(node).unwrap()
                };

                stack.push((next, w));
            }
        }

        acc
    }

    /// Batch prediction with optional excluded mask.
    ///
    /// If `excluded_mask` is None, this uses the fast deterministic `predict` path.
    /// If `excluded_mask` is Some, it uses `predict_excluded_mask`.
    pub fn predict_batch_excluded_mask(
        &self,
        X: &ArrayView2<'_, f64>,
        excluded_mask: Option<&[bool]>,
    ) -> Array1<f64> {
        let mut out = Array1::<f64>::zeros(X.nrows());

        match excluded_mask {
            None => {
                for (i, row) in X.outer_iter().enumerate() {
                    if let Some(slc) = row.as_slice() {
                        out[i] = self.predict(slc);
                    } else {
                        // fallback deterministic traversal
                        let mut node = 0usize;
                        loop {
                            if self.is_leaf(node) {
                                out[i] = self.value[node];
                                break;
                            }
                            let feat = self.feature[node];
                            let thr = self.threshold[node];
                            let xi = row[feat];
                            node = if xi.is_nan() || xi >= thr {
                                self.right_child(node).unwrap()
                            } else {
                                self.left_child(node).unwrap()
                            };
                        }
                    }
                }
            }
            Some(mask) => {
                for (i, row) in X.outer_iter().enumerate() {
                    if let Some(slc) = row.as_slice() {
                        out[i] = self.predict_excluded_mask(slc, mask);
                    } else {
                        out[i] = self.predict_excluded_row_view(&row, mask);
                    }
                }
            }
        }

        out
    }
    
    pub fn to_dump_parts(&self) -> TreeDumpParts {
        let n = self.value.len();

        let mut split_feature = Vec::<i32>::with_capacity(n);
        let mut split_value = Vec::<f64>::with_capacity(n);

        for i in 0..n {
            let is_leaf = self.leaf_id[i] >= 0;
            if is_leaf {
                split_feature.push(-1);
                split_value.push(0.0); // unused for leaves
            } else {
                split_feature.push(self.feature[i] as i32);
                split_value.push(self.threshold[i]);
            }
        }

        TreeDumpParts {
            split_feature,
            split_value,
            left_child: self.left_child.clone(),
            right_child: self.right_child.clone(),
            leaf_value: self.value.clone(),
            n_left: self.n_left.clone(),
            n_right: self.n_right.clone(),
            root_index: 0,
        }
    }

    #[cfg(debug_assertions)]
    pub fn validate_invariants(&self) {
        for i in 0..self.value.len() {
            let structural_leaf = self.is_leaf(i);
            let semantic_leaf = self.leaf_id[i] >= 0;
            debug_assert_eq!(
                structural_leaf,
                semantic_leaf,
                "leaf mismatch at node {i}"
            );
        }
    }
}
