# Third-party source

`sparkinfer/` is a pinned Git submodule from
<https://github.com/tpurtell/sparkinfer-glmrt>. GLMRT uses that source for
every Spark and coordinator CuTe AOT export; an independently installed
`b12x` or `sparkinfer` package is not a supported build input.

Initialize it after cloning GLMRT:

```bash
git submodule update --init --recursive third_party/sparkinfer
python3 scripts/verify-sparkinfer-source.py \
  --source third_party/sparkinfer \
  --lock third_party/sparkinfer.lock.json
```

The lock records both the fork commit and a deterministic content digest.
The digest keeps release archives verifiable after Git metadata is removed.
When intentionally updating the pin, update the submodule first, obtain the
new digest with `--print-tree-sha256`, update the lock, then run the full
verification command. The schema is:

```json
{
  "schema": 1,
  "repository": "https://github.com/tpurtell/sparkinfer-glmrt.git",
  "revision": "<lowercase 40-hex commit>",
  "source_tree_sha256": "<lowercase 64-hex digest>"
}
```

Generate the digest from a clean checkout; full verification also rejects a
wrong Git origin, a different `HEAD`, and tracked or non-ignored untracked
source changes. Never point a build at an unverified cache checkout.

`gptqmodel/` is a separately pinned submodule from the GLMRT GPTQModel fork.
It is a build input only for the reproducible EXL3 quantization image; neither
the coordinator nor Spark serving image imports GPTQModel at runtime. Verify it
after checkout with:

```bash
git submodule update --init --checkout third_party/gptqmodel
python3 scripts/verify-gptqmodel-source.py \
  --source third_party/gptqmodel \
  --lock third_party/gptqmodel.lock.json
```

The GPTQModel lock has the same revision/content-digest contract described
above. GLMRT intentionally pins the fork rather than resolving an arbitrary
PyPI or upstream Git revision during a quantization build.
