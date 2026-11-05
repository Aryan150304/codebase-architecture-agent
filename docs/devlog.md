# Devlog

One entry per observed problem: what went wrong, what changed, and what the change achieved.

## v0.1 — walking skeleton

### 2026-11-05 — The embedding model will not load under transformers 5

**Problem.** The first `reposcope ingest` run against Hadoop MapReduce chunked 2141 chunks from
332 files, then failed while constructing the model:

```
ImportError: cannot import name 'find_pruneable_heads_and_indices'
from 'transformers.pytorch_utils'
```

`jinaai/jina-embeddings-v2-base-code` is not a built-in architecture. It ships its own
`configuration_bert.py` and `modeling_bert.py`, which `trust_remote_code=True` allows
transformers to execute. That code imports an internal helper that transformers 5 removed.
`uv` had resolved transformers 5.22.1, which the model never supported.

**Change.** Pinned the supported range in `pyproject.toml` rather than working around it:
`transformers>=4.44,<5` and `sentence-transformers>=3.0,<4`, with a comment recording the reason
so a later dependency bump does not silently reintroduce it.

**Alternative rejected.** Switching to a model without remote code, such as
`BAAI/bge-base-en-v1.5`, would have removed this failure class entirely and kept transformers 5.
Its 512-token limit would have silently truncated the longest 40% of our chunks, so retrieval
quality would have degraded invisibly and the eval would measure a truncation artifact rather
than the chunking strategy. A loud `ImportError` is a better failure than silent truncation.

**Result.** The model loads. The debt is explicit: we still execute third-party code that can
break on any transformers upgrade. Moving to a native architecture is now a one-variable
experiment measurable against the v0.1 baseline.

### 2026-11-05 — GPU out of memory while embedding

**Problem.** With the model loading, `encode()` died on the very first batch:

```
torch.OutOfMemoryError: MPS command buffer execution failed:
Insufficient Memory ... Batches: 0%| | 0/67
```

The run passed all 2141 chunks to `encode()` at once with sentence-transformers' default batch
size of 32, and the model carries an 8192-token context. Attention memory grows with the square
of sequence length, so the longest chunk in a batch sets the cost for the whole batch. Because
sentence-transformers sorts longest-first, batch 0 is the worst case, which is why it failed
immediately instead of degrading. On an M1 with 8 GB shared between CPU and GPU, roughly 2 GB of
attention activations did not fit.

**Change.** Measured the chunks before choosing numbers. Over the ingested subtree: median 1325
characters, p99 2130, maximum 3549 (about 1200 tokens), with only 3 chunks above 3000
characters. Then set `MAX_SEQ_LENGTH = 2048` and `BATCH_SIZE = 8`, and exposed `--batch-size` on
the CLI.

**Why measure first.** Lowering `max_seq_length` to 512 would also have fixed the OOM, and would
have truncated most chunks without any error. The measurement is what makes a value that is both
cheap and lossless defensible. This couples chunk size to `MAX_SEQ_LENGTH`: raising
`--chunk-size` requires rechecking it, which the docstring states.

**Result.** 2141 chunks embedded and stored in 3m15s, 268 batches at 1.37 batches/second, no
memory errors.

### 2026-11-05 — First retrieval observations (no change yet)

The tools are in place, so semantic search could be run against the 2141 ingested chunks for the
first time. Recorded here as evidence for later versions; nothing was changed in response,
because v0.1 needs baseline eval numbers before any of it can be called an improvement.

Query: "how is a job submitted to the cluster", k=3.

1. `ClusterMetrics.java:181-220` — `getTotalJobSubmissions()`, a metrics getter. A false
   positive: it matched the words "job submissions" and "cluster" rather than the mechanism.
2. `Job.java:1651-1690` — `submit()` calling `submitter.submitJobInternal(Job.this, cluster)`.
   The correct answer, at rank 2.
3. `Job.java:1681-1720` — `waitForCompletion`, overlapping the previous hit by 10 lines.

Three things this suggests:

- **The correct chunk ranked second.** A single-search RAG baseline would answer this question
  from a metrics getter. This is the clearest motivation so far for the agent loop, and the
  baseline in step 7 should show it as a measured difference rather than an assumption.
- **`JobSubmitter.java` did not appear at all**, although it is the class that performs the
  submission, while `grep_code "class JobSubmitter"` located it immediately at line 70. Semantic
  search found the caller; lexical search found the implementation. Candidate trigger for the
  dependency-graph and `find_symbol` tools.
- **Adjacent chunks both surfaced**, so one of three result slots repeated lines the model
  already had. Overlap helps recall and costs context. Candidate triggers: deduplicating
  adjacent ranges, or symbol-level chunking so that chunk boundaries follow methods.
