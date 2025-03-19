
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.random as jrd


def offset_input(x_offset: jax.Array):
    def decorator(func):
        def wrapper(x, *args, **kwargs):
            return func(x + x_offset, *args, **kwargs)
        return wrapper
    return decorator


def add_noise(sigma: float):

    def decorator(func):
        def wrapper(x, key: jax.Array, *args, **kwargs):

            # Add gaussian noise
            noise = jax.random.normal(jrd.key(key.squeeze()), shape=()) * sigma

            kwargs.update({'noise': noise})

            return func(x, *args, **kwargs)
        return wrapper
    return decorator


def standard_scale_output(mean: float, sigma: float):
    def decorator(func):
        def wrapper(x, *args, **kwargs):
            kwargs.update({'mean': mean, 'sigma': sigma})
            return func(x, *args, **kwargs)
        return wrapper
    return decorator


def scale_input(domain_bounds: jax.Array, function_bounds: jax.Array):
    def decorator(func):
        def wrapper(x, *args, **kwargs):
            # Extract bounds
            dmn_lower = domain_bounds[:, 0]
            dmn_upper = domain_bounds[:, 1]
            fn_lower = function_bounds[:, 0]
            fn_upper = function_bounds[:, 1]

            # Scale x to [0, 1] based on standard bounds
            normalized_x = (x - dmn_lower) / (dmn_upper - dmn_lower)

            # Scale normalized_x to target bounds
            scaled_x = fn_lower + normalized_x * (fn_upper - fn_lower)

            # Pass transformed input to the original function
            return func(scaled_x, *args, **kwargs)
        return wrapper
    return decorator

# =============================================================================


def identity():
    def decorator(func):
        def wrapper(x, *args, **kwargs):
            return func(x, *args, **kwargs)
        return wrapper
    return decorator

# =============================================================================


def create_grad_noise(fn: Callable, sigma: float):
    # Forward pass
    def fn_fwd(x, key):
        primal_out = fn(x, key)
        return primal_out, (x, key)

    # Backward pass
    def fn_bwd(res, g):
        x, key = res
        noise = jax.random.normal(jrd.key(key.squeeze()), shape=x.shape) * sigma
        grad_out = jax.grad(fn)(x, key)

        tangent_out = noise + grad_out

        return (g * tangent_out, None)

    return fn_fwd, fn_bwd

# =============================================================================


def bench_function(fn_class: Callable,
                   seed: Optional[int],
                   scale_bounds: Optional[jax.Array],
                   dimensionality: Optional[int],
                   offset: bool,
                   noise: float,
                   scale_output: bool,
                   grad_noise: float,
                   ) -> Callable:
    fn = fn_class(dimensionality)

    if dimensionality is None:
        dimensionality = fn.dim_default

    # OFFSET
    if offset:
        # create random offset
        x_offset = jrd.uniform(jrd.key(seed), (dimensionality,))
        fn_offset = offset_input(x_offset)

    else:
        fn_offset = identity()

    # SCALE_BOUNDS
    if scale_bounds is None:
        fn_scale_bounds = identity()
    else:
        fn_scale_bounds = scale_input(scale_bounds, fn.bounds)

    # Wrap fn.evaluate to return f(x) and an empty kwargs dict
    def fn_with_kwargs(x, *args, noise: float, mean: float,
                       sigma: float, **kwargs) -> jax.Array:

        return ((fn.evaluate(x, *args) - mean) / sigma) + noise

    # Combine the input decorators
    fn_bench = fn_offset(fn_scale_bounds(fn_with_kwargs))

    # SCALE_OUTPUT
    if scale_output:
        # sample 100 points per dimension between the bounds
        no_noise = add_noise(0.0)
        no_scale_output = standard_scale_output(mean=0.0, sigma=1.0)
        _k = jrd.key(seed)
        key_scale_output = jrd.randint(key=_k, shape=(1,),
                                       minval=0, maxval=2**16)
        x = jax.random.uniform(key=_k, shape=(100, dimensionality))
        y = jax.vmap(partial(no_noise(no_scale_output(fn_bench)), key=key_scale_output))(x)
        fn_scale_output = standard_scale_output(mean=y.mean(), sigma=y.std())
    else:
        fn_scale_output = standard_scale_output(mean=0.0, sigma=1.0)

    # NOISE
    fn_noise = add_noise(noise)

    if grad_noise == 0.0:
        return fn_noise(fn_scale_output(fn_bench))
    else:
        fn_grad_noise = jax.custom_vjp(fn_noise(fn_scale_output(fn_bench)))
        fn_grad_noise.defvjp(*create_grad_noise(
            fn=fn_noise(fn_scale_output(fn_bench)), sigma=grad_noise))
        return fn_grad_noise
