//   Copyright 2024 The PyMC Developers
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
#![warn(missing_docs)]
#![allow(non_snake_case)]

//! pg_bart provides an extensible implementation of Bayesian Additive
//! Regression Trees (BART). BART is a non-parametric method to
//! approximate functions based on the sum of many trees where
//! priors are used to regularize inference, mainly by restricting
//! a tree's learning capacity so that no individual tree is able
//! to explain the data, but rather the sum of trees. Inference is
//! performed using a sampler inspired by the Particle Gibbs method
//! introduced by Lakshminarayanan et al. [2015].

pub mod data;
pub mod math;
pub mod ops;
pub mod particle;
pub mod pgbart;
pub mod split_rules;
pub mod tree;

use crate::data::ExternalData;
use crate::ops::Response;
use crate::pgbart::{PgBartSettings, PgBartState};
use crate::split_rules::{ContinuousSplit, OneHotSplit, SplitRuleType};
use crate::tree::{DecisionTree, deserialize_forest, serialize_forest};

use std::str::FromStr;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use ndarray::{ArrayD, IxDyn};
use numpy::PyArrayDyn;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use pyo3::types::{PyDict, PyList};
use pyo3::types::PyBytes;

/// `StateWrapper` wraps around `PgBartState` to hold state pertaining to
/// the Particle Gibbs sampler and posterior draws.
///
/// This class is `unsendable`, i.e., it cannot be sent across threads safely.
#[pyclass(unsendable)]
struct StateWrapper {
    state: Option<PgBartState>,
    // Posterior draws stored in Rust:
    draws: Vec<Vec<DecisionTree>>,
}

#[pymethods]
impl StateWrapper {
    fn export_all_trees<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty_bound(py); // draws

        for draw in &self.draws {
            let py_ensemble = PyList::empty_bound(py); // trees

            for tree in draw {
                let dump = tree.to_dump_parts();

                let d = PyDict::new_bound(py);
                d.set_item("split_feature", dump.split_feature)?;
                d.set_item("split_value", dump.split_value)?;
                d.set_item("left_child", dump.left_child)?;
                d.set_item("right_child", dump.right_child)?;
                d.set_item("leaf_value", dump.leaf_value)?;
                d.set_item("n_left", dump.n_left)?;
                d.set_item("n_right", dump.n_right)?;
                d.set_item("root_index", dump.root_index)?;

                py_ensemble.append(d)?;
            }

            out.append(py_ensemble)?;
        }

        Ok(out)
    }

    /// Export a single draw (full forest) as a compact byte blob for IPC.
    fn export_draw_as_bytes<'py>(&self, py: Python<'py>, draw_idx: usize) -> PyResult<Bound<'py, PyBytes>> {
        let draw = self.draws.get(draw_idx).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Draw index {draw_idx} out of bounds (n_draws={}).",
                self.draws.len()
            ))
        })?;

        let bytes = serialize_forest(draw);
        Ok(PyBytes::new_bound(py, &bytes))
    }

    /// Load all draws from byte blobs produced by `export_draw_as_bytes`.
    /// Python owns the bytes; Rust owns the decoded trees after this call.
    fn load_all_trees_from_bytes(&mut self, draws: Vec<Vec<u8>>) -> PyResult<()> {
        let mut decoded = Vec::with_capacity(draws.len());

        for (idx, bytes) in draws.into_iter().enumerate() {
            let forest = deserialize_forest(&bytes).map_err(|err| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to decode draw {idx}: {err}"
                ))
            })?;
            decoded.push(forest);
        }

        self.draws = decoded;
        Ok(())
    }
}



#[pyclass]
#[derive(Clone)]
struct TreeDump {
    #[pyo3(get)]
    split_feature: Vec<i32>,
    #[pyo3(get)]
    split_value: Vec<f64>,
    #[pyo3(get)]
    left_child: Vec<i32>,
    #[pyo3(get)]
    right_child: Vec<i32>,
    #[pyo3(get)]
    leaf_value: Vec<f64>,
    #[pyo3(get)]
    n_left: Vec<i32>,
    #[pyo3(get)]
    n_right: Vec<i32>,
    #[pyo3(get)]
    root_index: i32,
}


#[pyfunction]
#[pyo3(signature = (
    X,
    y,
    logp,
    alpha,
    beta,
    split_prior,
    split_rules,
    response,
    n_trees,
    n_particles,
    leaf_sd,
    batch,
    leaves_shape,
    seed=None,
))]
#[allow(clippy::too_many_arguments)]
fn initialize(
    X: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    logp: usize,
    alpha: f64,
    beta: f64,
    split_prior: PyReadonlyArray1<f64>,
    split_rules: Vec<String>,
    response: String,
    n_trees: usize,
    n_particles: usize,
    leaf_sd: Vec<f64>,
    batch: (f64, f64),
    leaves_shape: usize,
    seed: Option<u64>,
) -> PyResult<StateWrapper> {
    // Heap allocation because size of 'ExternalData' is not known at compile time
    let data = Box::new(ExternalData::new(X, y, logp));
    let response = Response::from_str(&response).unwrap();
    let mut rules: Vec<SplitRuleType> = Vec::new();

    for rule in split_rules {
        let split = match rule.as_str() {
            "ContinuousSplit" => SplitRuleType::Continuous(ContinuousSplit),
            "OneHotSplit" => SplitRuleType::OneHot(OneHotSplit),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown split type: {}",
                    rule
                )))
            }
        };
        rules.push(split);
    }

    let params = PgBartSettings::new(
        n_trees,
        n_particles,
        alpha,
        beta,
        leaf_sd,
        batch,
        split_prior.to_vec().unwrap(),
        response,
        rules,
        leaves_shape,
    );
    let rng_seed = seed.unwrap_or_else(|| StdRng::from_entropy().gen());
    let state = PgBartState::new(params, data, rng_seed);

    Ok(StateWrapper { state: Some(state), draws: Vec::new() })
}

#[pyfunction]
fn make_predictor() -> PyResult<StateWrapper> {
    Ok(StateWrapper {
        state: None,
        draws: Vec::new(),
    })
}

#[pyfunction]
fn step<'py>(
    py: Python<'py>,
    wrapper: &mut StateWrapper,
    tune: bool,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
    Vec<TreeDump>,
) {
    // Get mutable access to the sampler state
    let state = wrapper.state.as_mut().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "This StateWrapper has no sampler state (prediction-only). \
             Use initialize(...) to create a sampler state."
        )
    }).unwrap(); // if you prefer, propagate instead of unwrap

    // Update tune flag
    state.tune = tune;

    // Run sampler
    state.step();

    // Record posterior draw in Rust only when not tuning
    if !tune {
        let draw: Vec<DecisionTree> = state.trees().cloned().collect();
        wrapper.draws.push(draw);
    }

    // Predictions
    let predictions = state.predictions();
    let py_preds_array = PyArray1::from_array_bound(py, &predictions.view());

    // Variable inclusion
    let variable_inclusion = state.variable_inclusion().clone();
    let py_variable_inclusion_array = PyArray1::from_vec_bound(py, variable_inclusion);

    let tree_dumps: Vec<TreeDump> = Vec::new();
    (py_preds_array, py_variable_inclusion_array, tree_dumps)
}

#[pyfunction]
#[pyo3(signature = (wrapper, X, size=None, excluded=None, shape=1, seed=None))]
#[allow(clippy::too_many_arguments)]
fn sample_posterior<'py>(
    py: Python<'py>,
    wrapper: &StateWrapper,
    X: PyReadonlyArray2<f64>,
    size: Option<Vec<usize>>,
    excluded: Option<Vec<usize>>,
    shape: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    // We are assuming separate_trees = False AND scalar leaf values.
    // That only produces correct results when shape == 1.
    if shape != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Rust sample_posterior currently supports scalar leaf values only (shape must be 1). \
             For vector-valued outputs, set separate_trees=True or implement vector-valued leaves.",
        ));
    }

    let n_draws = wrapper.draws.len();
    if n_draws == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No posterior draws stored in Rust yet. Call step(..., tune=False) at least once before sample_posterior().",
        ));
    }

    // Resolve size_iter like Python:
    // - None -> (1,)
    // - int -> [int]  (we accept Vec so Python wrapper should pass [int])
    // - tuple -> Vec
    let size_iter: Vec<usize> = match size {
        None => vec![1],
        Some(v) if v.is_empty() => vec![1],
        Some(v) => v,
    };

    let flatten_size: usize = size_iter.iter().product();

    // Borrow X as an ndarray view (no copy)
    let x_view = X.as_array();
    let n_obs = x_view.nrows();
    let n_features = x_view.ncols();

    // Build excluded mask once for speed (avoids O(k) scans inside traversal)
    let excluded_mask: Option<Vec<bool>> = excluded.as_ref().map(|ex| {
        let mut mask = vec![false; n_features];
        for &idx in ex {
            if idx < n_features {
                mask[idx] = true;
            }
        }
        mask
    });

    // Output shape matches Python reshape: (*size_iter, n_obs, shape)
    let mut out_shape: Vec<usize> = Vec::with_capacity(size_iter.len() + 2);
    out_shape.extend_from_slice(&size_iter);
    out_shape.push(n_obs);
    out_shape.push(shape); // == 1

    // Compute in Rust (keep GIL; avoids Ungil/Sync issues because wrapper contains PyData)
    let seed_val = seed.unwrap_or(0xD1CE_BA5Eu64);
    let mut rng = StdRng::seed_from_u64(seed_val);

    // Allocate output contiguous buffer
    let mut out_arr = ArrayD::<f64>::zeros(IxDyn(&out_shape));
    let out_slice = out_arr
        .as_slice_mut()
        .expect("output should be contiguous");

    // Precompute indices into draws
    let mut draw_idx = Vec::with_capacity(flatten_size);
    for _ in 0..flatten_size {
        draw_idx.push(rng.gen_range(0..n_draws));
    }

    let stride_sample = n_obs * shape; // shape == 1

    for (s, &d) in draw_idx.iter().enumerate() {
        let base = s * stride_sample;

        for tree in &wrapper.draws[d] {
            let preds = tree.predict_batch_excluded_mask(
                &x_view,
                excluded_mask.as_ref().map(|m| m.as_slice()),
            );

            for i in 0..n_obs {
                out_slice[base + i] += preds[i];
            }
        }
    }

    let out = out_arr;



    // Convert to NumPy array without extra copies
    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}


#[pymodule]
fn pymc_bart_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(initialize, m)?)?;
    m.add_function(wrap_pyfunction!(step, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(sample_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(make_predictor, m)?)?;
    m.add_class::<TreeDump>()?;

    Ok(())
}
