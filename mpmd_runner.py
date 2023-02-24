import argparse

import numpy as np
from typing import Any, Callable, Tuple, Union, Iterable, Sequence, List

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
from flax.linen import partitioning
from flax.training import train_state

import optax

PRNGKey = Any
Shape = Tuple[int, ...]
DType = jnp.dtype
Array = jnp.ndarray
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision,
                                                                       lax.Precision]]
Initializer = Callable[[PRNGKey, Shape, DType], Array]


PARAMS_KEY = 'params'
PARAMS_AXES_KEY = PARAMS_KEY + '_axes'

DP_SIZE = 8
BATCH_PER_GPU = 32
GLOBAL_BATCH = BATCH_PER_GPU * DP_SIZE
SEQLEN = 512
HIDDEN = 1024

def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)

def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return (x,)

class DenseGeneral(nn.Module):
    features: Union[Iterable[int], int]
    kernel_init: Initializer = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
    kernel_axes: Tuple[str, ...] = ('h1', 'h2')
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.bfloat16

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
        kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),) + features
        kernel = partitioning.param_with_axes('kernel',
                                                 self.kernel_init,
                                                 kernel_param_shape,
                                                 jnp.float32,
                                                 axes=self.kernel_axes)

        kernel = jnp.reshape(kernel, kernel_shape)

        contract_ind = tuple(range(0, len(axis)))

        kernel = jnp.asarray(kernel, self.dtype)
        y = lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))

        return y

encoder_lyr = DenseGeneral(features=1024)

def get_synthesis_data(data_rng, batch):
    return jax.random.normal(data_rng, [SEQLEN, batch, HIDDEN], jnp.float32)

def get_params_pspec(rules, params_axes):
    mapping_rules = {param_axis : mesh_axis for param_axis, mesh_axis in rules}
    params_pspec = jax.tree_map(lambda x: PartitionSpec(*(mapping_rules[key] for key in x)), params_axes)

    return params_pspec

def get_state_pspec(state, rules, params_axes):

    def replace_params(x):
        return params_pspec if isinstance(x, FrozenDict) else None

    params_pspec = get_params_pspec(rules, params_axes)
    state_pspec = jax.tree_map(replace_params,
                                state,
                                is_leaf=lambda x: isinstance(x, FrozenDict))
    return state_pspec

def train_step(batch, state, others):
    def loss_fn(collections):
        logits = encoder_lyr.apply(collections, batch)
        loss = jnp.mean(logits)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(FrozenDict({PARAMS_KEY:state.params, **others}))
    grads, params_grads = grads.pop(PARAMS_KEY)
    state = state.apply_gradients(grads=params_grads)
    return loss, state, others

def main(args):

    device_mesh = mesh_utils.create_device_mesh((DP_SIZE,))
    dp_mesh_axis_name = 'data'

    with Mesh(devices=device_mesh, axis_names=(dp_mesh_axis_name,)):
        rng = jax.random.PRNGKey(0)
        rng, init_rng, data_rng = jax.random.split(rng, 3)

        inputs = get_synthesis_data(data_rng, GLOBAL_BATCH)
        rules = (('h1', None), ('h2', None))

        abstract_variables = jax.eval_shape(encoder_lyr.init, init_rng, inputs)
        params_axes = partitioning.get_axis_names(abstract_variables[PARAMS_AXES_KEY])

        params_pspec = get_params_pspec(rules, params_axes)
        inputs_pspec = PartitionSpec(None, dp_mesh_axis_name, None)

        with partitioning.axis_rules(rules):
            in_shardings = (None, inputs_pspec)
            out_shardings = FrozenDict({key: params_pspec if key is PARAMS_KEY else None \
                                        for key in abstract_variables})
            variables = pjit(encoder_lyr.init, in_shardings, out_shardings)(init_rng, inputs)

        optimizer = optax.sgd(0.001, 0.9)
        variables, params = variables.pop(PARAMS_KEY)
        state = train_state.TrainState.create(apply_fn=encoder_lyr.apply,
                                                params=params,
                                                tx=optimizer)

        state_pspec = get_state_pspec(state, rules, params_axes)
        pjitted_train_step = pjit(train_step,
                            (inputs_pspec, state_pspec, None),
                            (None, state_pspec, None))

        for i in range(5):
            rng, data_rng = jax.random.split(rng)
            inputs = get_synthesis_data(data_rng, GLOBAL_BATCH)
            loss, state, variables = pjitted_train_step(inputs, state, variables)
            if args.process_id == 0:
                print(f"Step {i} - Loss: {loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinator_address',
                        type=str,
                        default='127.0.0.1:1234')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1)
    parser.add_argument('--process_id',
                        type=int,
                        default=0)
    args = parser.parse_args()

    jax.distributed.initialize(coordinator_address=args.coordinator_address,
                               num_processes=args.num_processes,
                               process_id=args.process_id,
                               local_device_ids=args.process_id)
    main(args)

