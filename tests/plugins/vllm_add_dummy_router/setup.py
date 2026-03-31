# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="vllm_add_dummy_router",
    version="0.1",
    packages=["vllm_add_dummy_router"],
    entry_points={
        "vllm.router_plugins": ["dummy_router = vllm_add_dummy_router:attach_router"]
    },
)
