
from typing import Callable, Optional

import jax
import jax.random as jrd


def offset_input(x_offset: jax.Array):
    def decorator(func):
        def wrapper(x):
            return func(x + x_offset)
        return wrapper
    return decorator

# Inverse of the offset_input decorator (subtract the offset)


def inverse_offset_input(x_offset: jax.Array):
    def decorator(func):
        def wrapper(x):
            return func(x - x_offset)  # Subtract the offset
        return wrapper
    return decorator


# Add noise decorator with dynamic RNG handling and noise level as a percentage of y
def add_noise(noise_level: float, rng_key):
    rng_container = [rng_key]  # Use a mutable container to store the RNG

    def decorator(func):
        def wrapper(x):
            # Compute the original (noiseless) y value
            y_noiseless = func(x)

            # Update the RNG key
            rng_container[0], subkey = jax.random.split(rng_container[0])

            # Calculate the noise as a percentage of the noiseless y value
            noise = jax.random.normal(subkey, shape=()) * noise_level * y_noiseless

            return y_noiseless + noise  # Return noisy output
        return wrapper
    return decorator

# Inverse of the add_noise decorator
# Note: This assumes that we want to approximate the original value by removing noise.
# Since noise is random, perfect inversion isn't possible, but we can try to "undo" the scaling.


def inverse_add_noise(noise_level: float, rng_key):
    rng_container = [rng_key]  # Use a mutable container to store the RNG

    def decorator(func):
        def wrapper(x):
            # Compute the noisy y value
            y_noisy = func(x)

            # The inverse operation would subtract the noise
            # This is an approximation since the noise is random and cannot be recovered
            # However, we can compute the expected noiseless value by removing the scaled noise.
            rng_container[0], subkey = jax.random.split(rng_container[0])
            noise = jax.random.normal(subkey, shape=()) * noise_level * y_noisy
            y_noiseless = y_noisy - noise  # Approximate the noiseless value

            return y_noiseless  # Return the approximated noiseless output
        return wrapper
    return decorator

# Inverse of the scale_input decorator


def inverse_scale_input(domain_bounds: jax.Array, function_bounds: jax.Array):
    def decorator(func):
        def wrapper(x):
            # Extract bounds
            dmn_lower = domain_bounds[:, 0]
            dmn_upper = domain_bounds[:, 1]
            fn_lower = function_bounds[:, 0]
            fn_upper = function_bounds[:, 1]

            # Scale x back from target bounds to [0, 1] based on the function bounds
            normalized_x = (x - fn_lower) / (fn_upper - fn_lower)

            # Scale normalized_x back to original domain bounds
            original_x = dmn_lower + normalized_x * (dmn_upper - dmn_lower)

            # Pass transformed input to the original function
            return func(original_x)
        return wrapper
    return decorator

# Scale input decorator with standard and target bounds


def scale_input(domain_bounds: jax.Array, function_bounds: jax.Array):
    def decorator(func):
        def wrapper(x):
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
            return func(scaled_x)
        return wrapper
    return decorator

# =============================================================================


def identity():
    def decorator(func):
        def wrapper(x):
            return func(x)
        return wrapper
    return decorator


def bench_function(fn_class: Callable,
                   seed: Optional[int] = None,
                   scale_bounds: Optional[jax.Array] = None,
                   dimensionality: Optional[int] = None,
                   offset: bool = False,
                   noise: float = 0.0,
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

    # NOISE
    if noise == 0.0:
        fn_noise = identity()
    else:
        fn_noise = add_noise(noise, jrd.key(seed))

    # SCALE_BOUNDS
    if scale_bounds is None:
        fn_scale_bounds = identity()
    else:
        fn_scale_bounds = scale_input(scale_bounds, fn.bounds)

    return fn_offset(fn_noise(fn_scale_bounds(fn.evaluate)))
