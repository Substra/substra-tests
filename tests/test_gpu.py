import pytest

from substratest.factory import FunctionCategory
from substratest.fl_interface import OutputIdentifiers


@pytest.mark.skip("Need an environment with GPUs")
def test_gpu(factory, client, org_idx, default_datasets, workers):
    """Test that the task can see the GPU"""
    org_idx = 0
    nvidia_drivers = "nvidiacuda11.8.0-base-ubuntu22.04"

    # Need the base image, the minimal image does not have pip
    dockerfile = """
FROM python:3.12-slim

RUN apt-get update -y && apt-get install -y git
RUN python3 -m pip install -U pip
RUN python3 -m pip install git+https://github.com/Substra/substra-tools.git@{substratools_git_ref}
RUN python3 -m pip install --no-cache-dir torch==2.0.1
COPY function.py .

ENTRYPOINT ["python3", "function.py"]
"""
    script = f"""
import json
import substratools as tools
import torch

@tools.register
def train(inputs, outputs, task_properties):
    assert torch.cuda.is_available()
    save_model(['test'], outputs['{OutputIdentifiers.shared}'])

@tools.register
def predict(X, model):
    assert torch.cuda.is_available()
    res = [x * model['value'] for x in X]
    print(f'Predict, get X: {{X}}, model: {{model}}, return {{res}}')
    return res

def load_model(path):
    with open(path) as f:
        return json.load(f)

def save_model(model, path):
    with open(path, 'w') as f:
        return json.dump(model, f)

if __name__ == '__main__':
    tools.execute()
"""  # noqa
    spec = factory.create_function(FunctionCategory.simple, dockerfile=dockerfile, py_script=script)
    function = client.add_function(spec)
    cp_spec = factory.create_compute_plan(tag=f"GPU test - org {default_datasets[org_idx].owner} - {nvidia_drivers}")
    cp_spec.create_traintask(
        function=function,
        inputs=default_datasets[org_idx].train_data_inputs,
        metadata={"docker_cuda_version": nvidia_drivers},
        workers=workers[org_idx],
    )
    cp_added = client.add_compute_plan(cp_spec)
    client.wait_compute_plan(cp_added.key)
