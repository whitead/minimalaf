# Minimal AlphaFold


Install:

```bash
pip install -r requirements.txt
modal deploy af.py
```


Usage from python (also in `run.py`):
```py
import modal

fasta = """\
>A chain
MTEYKLVVVGAGGVGKSALTIQLIQNHKLRKLNPPDESGPGCMNCKCVIS"""

af = modal.Function.lookup("minimalaf", "fold")
result = af.remote(fasta=fasta, 
          models=[1],
          num_recycles=1)
with open("results.zip", "wb") as f:
    f.write(result)
```

## Copyright

CC0. No Rights Reserved
