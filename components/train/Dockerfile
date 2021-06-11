# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#FROM nvcr.io/nvidia/tensorflow:19.03-py3
FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3

RUN pip install keras==2.2.4
RUN pip install --upgrade pip
#RUN pip install tf-nightly-gpu-2.0-preview
ADD src /workspace
WORKDIR /workspace

ENTRYPOINT ["python", "train.py"]

