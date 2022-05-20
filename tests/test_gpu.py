import pytest

from substratest.factory import AlgoCategory


@pytest.mark.skip("Need an environment with GPUs")
def test_gpu(factory, client, org_idx, default_datasets):
    """Test that the task can see the GPU"""
    org_idx = 0
    nvidia_drivers = "nvidiacuda11.6.0-base-ubuntu20.04"

    # Need the base image, the minimal image does not have pip
    dockerfile = f"""
FROM gcr.io/connect-314908/connect-tools:latest-{nvidia_drivers}-python3.7

RUN python3 -m pip install torch==1.11.0
COPY algo.py .

ENTRYPOINT ["python3", "algo.py"]
"""
    script = f"""
import json
import substratools as tools
import torch

class TestAlgo(tools.Algo):
    def train(self, X, y, models, rank):
        assert torch.cuda.is_available()
        return ['test']

    def predict(self, X, model):
        assert torch.cuda.is_available()
        res = [x * model['value'] for x in X]
        print(f'Predict, get X: {{X}}, model: {{model}}, return {{res}}')
        return res

    def load_model(self, path):
        with open(path) as f:
            return json.load(f)

    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)

if __name__ == '__main__':
    tools.algo.execute(TestAlgo())
"""  # noqa
    spec = factory.create_algo(AlgoCategory.simple, dockerfile=dockerfile, py_script=script)
    algo = client.add_algo(spec)
    cp_spec = factory.create_compute_plan(tag=f"GPU test - org {default_datasets[org_idx].owner} - {nvidia_drivers}")
    cp_spec.create_traintuple(
        algo=algo,
        dataset=default_datasets[org_idx],
        data_samples=default_datasets[org_idx].train_data_sample_keys,
        metadata={"docker_cuda_version": nvidia_drivers},
    )
    cp_added = client.add_compute_plan(cp_spec)
    client.wait(cp_added)
