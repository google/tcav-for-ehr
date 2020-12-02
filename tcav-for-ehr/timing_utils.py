# Copyright 2020 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
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
