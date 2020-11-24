# Lint as: python3
"""Utilities for timing code executions."""

import contextlib
import timeit

from absl import logging


@contextlib.contextmanager
def time_this(name, logger=logging.info):
  """Context manager for logging execution time."""
  start_time = timeit.default_timer()
  yield
  duration = timeit.default_timer() - start_time
  logger("Execution of `{}` took {} seconds".format(name, duration))
