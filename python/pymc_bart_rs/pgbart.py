#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from time import perf_counter
from typing import Optional, Tuple

import numpy as np

from pymc.initial_point import PointType
from pymc.model import Model, modelcontext
from pymc.pytensorf import inputvars
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.compound import Competence
from pytensor.graph.basic import Variable

from pymc_bart_rs.bart import BARTRV
from pymc_bart_rs.compile_pymc import CompiledPyMCModel
from pymc_bart_rs.pymc_bart_rs import initialize, step
#from pymc_bart_rs.tree_dump import TreeDump


class PGBART(ArrayStepShared):
    """
    Particle Gibss BART sampling step.

    Parameters
    ----------
    vars: list
        List of value variables for sampler
    num_particles : tuple
        Number of particles. Defaults to 10
    batch : tuple
        Number of trees fitted per step. The first element is the batch size during tuning and the
        second the batch size after tuning.  Defaults to  (0.1, 0.1), meaning 10% of the `m` trees
        during tuning and after tuning.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """

    name = "pgbart"
    default_blocked = False
    generates_stats = True
    stats_dtypes_shapes: dict[str, tuple[type, list]] = {
        "variable_inclusion": (object, []),
        "tune": (bool, []),
        "time": (float, []),
    }

    def __init__(  # noqa: PLR0915
        self,
        vars=None,  # pylint: disable=redefined-builtin
        num_particles: int = 10,
        batch: Tuple[float, float] = (0.1, 0.1),
        model: Optional[Model] = None,
        initial_point: PointType | None = None,
        compile_kwargs: dict | None = None,  # pylint: disable=unused-argument
    ):
        model = modelcontext(model)
        if initial_point is None:
            initial_point = model.initial_point()

        # Get the instances of the BART random variable from the PyMC model
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)
        value_bart = vars[0]
        self.bart = model.values_to_rvs[value_bart].owner.op

        # Auto-reset shared draw storage at the start of every pm.sample() run.
        # This runs in the main process before multiprocessing workers start.
        if hasattr(self.bart, "_reset_posterior_cache"):
            self.bart._reset_posterior_cache()
        else:
            # Fallback: best-effort clear
            try:
                while len(self.bart.all_trees):
                    self.bart.all_trees.pop()
            except Exception:
                pass
            self.bart._python_all_trees = None
            self.bart._rust_draws_loaded = False
            self.bart._rust_state = None

        if isinstance(self.bart.X, Variable):
            self.X = self.bart.X.eval()
        else:
            self.X = self.bart.X

        if isinstance(self.bart.Y, Variable):
            self.Y = self.bart.Y.eval()
        else:
            self.Y = self.bart.Y

        self.m = self.bart.m
        self.response = self.bart.response
        self.num_particles = num_particles
        self.batch = batch

        shape = initial_point[value_bart.name].shape
        self.shape = 1 if len(shape) == 1 else shape[0]

        print(f"shape: {shape}, self.shape: {self.shape}")

        # Set trees_shape (dim for separate tree structures)
        # and leaves_shape (dim for leaf node values)
        # One of the two is always one, the other equal to self.shape
        self.trees_shape = self.shape if self.bart.separate_trees else 1
        self.leaves_shape = self.shape if not self.bart.separate_trees else 1

        if self.bart.split_prior.size == 0:
            self.alpha_vec = np.ones(self.X.shape[1])
        else:
            self.alpha_vec = self.bart.split_prior

        if self.bart.split_rules:
            self.split_rules = self.bart.split_rules
        else:
            self.split_rules = ["ContinuousSplit"] * self.X.shape[1]

        # If data is binary
        # self.leaf_sd = np.ones((self.trees_shape, self.leaves_shape))
        self.leaf_sd = np.ones(self.leaves_shape)

        y_unique = np.unique(self.bart.Y)
        if y_unique.size == 2 and np.all(y_unique == [0, 1]):
            self.leaf_sd *= 3 / self.m**0.5
        else:
            self.leaf_sd *= self.bart.Y.std() / self.m**0.5

        # Compile the PyMC model to create a C callback. This function pointer is
        # passed to Rust and called using Rust's foreign function interface (FFI)
        self.compiled_pymc_model = CompiledPyMCModel(model, vars)

        # Rust sampler state is created lazily in set_rng so seeds propagate from PyMC
        self.state = None
        self._rust_seed = None

        self.tune = True
        self.draw_idx = 0
        super().__init__(vars, self.compiled_pymc_model.shared)

        # Reset draw index for this chain
        self.draw_idx = 0
    
    def set_rng(self, rng):
        """PyMC provides a per-chain Generator so we can derive deterministic seeds."""
        super().set_rng(rng)

        seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        self._rust_seed = seed

        self.state = initialize(
            X=self.X,
            y=self.bart.Y,
            logp=self.compiled_pymc_model.get_function_pointer(),
            alpha=self.bart.alpha,
            beta=self.bart.beta,
            split_prior=self.alpha_vec,
            split_rules=self.split_rules,
            response=self.bart.response,
            n_trees=self.bart.m,
            n_particles=self.num_particles,
            leaf_sd=self.leaf_sd,
            batch=self.batch,
            leaves_shape=self.leaves_shape,
            seed=seed,
        )
        self.draw_idx = 0

    def astep(self, _):
        if self.state is None:
            raise RuntimeError(
                "Rust sampler state is not initialized. PyMC should call set_rng() "
                "before sampling; if you hit this, something is wrong with step initialization."
            )

        #print("PGBART.astep tune =", self.tune)
        # Record time to quantify performance improvements
        t0 = perf_counter()
        self.compiled_pymc_model.update_shared_arrays()

        sum_trees, variable_inclusion, _ = step(self.state, self.tune)
        #print("after step; tune =", self.tune, "has _rust_state?", hasattr(self.bart, "_rust_state"))

        if not self.tune:
            draw_bytes = self.state.export_draw_as_bytes(self.draw_idx)
            #if self.draw_idx == 0:
            #    print("export_draw_as_bytes type:", type(draw_bytes))
            self.draw_idx += 1
            self.bart.all_trees.append(draw_bytes)

        t1 = perf_counter()

        stats = {
            "variable_inclusion": variable_inclusion,
            "tune": self.tune,
            "time": t1 - t0,
        }
        return sum_trees, [stats]

    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE
