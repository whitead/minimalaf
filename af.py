import os
import argparse

from modal import Image, App

app = App("minimalaf")

image = (
    # Image.from_registry("nvidia/cuda:12.4.99-runtime-ubuntu22.04", add_python="3.11")
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold"
    )
    .micromamba_install(
        "kalign2=2.04", "hhsuite=3.3.0", channels=["conda-forge", "bioconda"]
    )
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a100",
    )
    .run_commands("python -m colabfold.download")
)

with image.imports():
    from colabfold.batch import get_queries, run
    from colabfold.download import default_data_dir


@app.function(image=image, gpu='a10g')
def fold(fasta: str, models: list[int] = [1], num_recycles=3):
    out_dir = "/tmp"  # noqa: S108
    with open("input_sequence.fasta", "w") as f:
        f.write(fasta)
    queries, is_complex = get_queries(".")
    run(
        queries=queries,
        result_dir=out_dir,
        use_templates=False,
        num_relax=0,
        relax_max_iterations=200,
        msa_mode="MMseqs2 (UniRef+Environmental)",
        model_type="auto",
        num_models=len(models),
        num_recycles=num_recycles,
        model_order=models,
        is_complex=is_complex,
        data_dir=default_data_dir,
        keep_existing_results=False,
        rank_by="auto",
        pair_mode="unpaired+paired",
        stop_at_score=100,
        zip_results=True,
        user_agent="colabfold/google-colab-batch",
    )
    # return the zip result
    path = next(f for f in os.listdir(out_dir) if f.endswith(".zip"))
    with open(os.path.join(out_dir, path), "rb") as g:
        return g.read()



