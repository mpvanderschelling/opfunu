
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
        def wrapper(x, key: int, *args, **kwargs):

            rng_key = jax.random.key(key)
            _, subkey = jax.random.split(jax.random.key(key))

            # Add gaussian noise
            noise = jax.random.normal(rng_key, shape=()) * sigma

            new_key = jax.random.randint(key=rng_key,
                                         minval=0,
                                         maxval=2**16,
                                         shape=(),
                                         )

            kwargs.update({'key': new_key.item(), 'noise': noise})

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


def bench_function(fn_class: Callable,
                   seed: Optional[int] = None,
                   scale_bounds: Optional[jax.Array] = None,
                   dimensionality: Optional[int] = None,
                   offset: bool = False,
                   noise: float = 0.0,
                   scale_output: bool = False,
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
    def fn_with_kwargs(x, *args, noise: float = 0.0, mean: float = 0.0,
                       sigma: float = 1.0, **kwargs) -> Tuple[jax.Array, Dict[str, Any]]:

        return ((fn.evaluate(x, *args) + noise) - mean) / sigma, kwargs

    # Combine the input decorators
    fn_bench = fn_offset(fn_scale_bounds(fn_with_kwargs))

    # SCALE_OUTPUT
    if scale_output:
        # sample 100 points per dimension between the bounds
        key_scale_output = jrd.key(seed)
        x = jax.random.uniform(key_scale_output, (100, dimensionality))
        y, _ = jax.vmap(partial(fn_bench, key=key_scale_output))(x)
        fn_scale_output = standard_scale_output(mean=y.mean(), sigma=y.std())
    else:
        fn_scale_output = identity()

    # NOISE
    if noise == 0.0:
        fn_noise = identity()
    else:
        fn_noise = add_noise(noise)

    return fn_noise(fn_scale_output(fn_bench))
