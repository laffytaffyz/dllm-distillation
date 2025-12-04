# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from . import transformers
from .auto import build_foundation_model, build_processor, build_tokenizer
from .module_utils import init_empty_weights, load_model_weights, save_model_assets, save_model_weights

# Lazy import seed_omni to avoid PIL/PILImageResampling import errors when not needed
# (seed_omni requires PIL which may not be installed for text-only models)
def __getattr__(name):
    if name in ("build_omni_model", "build_omni_processor"):
        from .seed_omni import build_omni_model, build_omni_processor
        if name == "build_omni_model":
            return build_omni_model
        elif name == "build_omni_processor":
            return build_omni_processor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_foundation_model",
    "build_omni_model",  # Lazy import via __getattr__
    "build_omni_processor",  # Lazy import via __getattr__
    "build_processor",
    "build_tokenizer",
    "init_empty_weights",
    "load_model_weights",
    "save_model_assets",
    "save_model_weights",
    "transformers",
]
