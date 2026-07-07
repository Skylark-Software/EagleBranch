# Working in this repository (EagleBranch)

**This is Skylark Software's fork of llama.cpp.** The AI-usage policy in this
file and in [CONTRIBUTING.md](CONTRIBUTING.md) applies to *this fork*. The
upstream llama.cpp project has its own, much more restrictive policy — it
applies to contributions **to upstream**, not to work here (see below).

## Skylark's AI policy for this fork

Skylark Software develops with AI openly. Much of this fork's distinguishing
work — the EAGLE3-DS architecture, kernel work, and documentation — was built
by or with AI, under human direction and review, and is attributed honestly
in the commit history. We consider disclosed, reviewed AI work a normal
engineering practice, not an exception.

For AI agents working in this repo:

- You may read, write, and refactor code and documentation.
- Attribute your work honestly (commit author or `Co-Authored-By`), per the
  repo owner's direction.
- The same quality bar applies to you as to anyone: the person merging must
  be able to explain and defend the change. Don't submit what can't be
  explained.
- Do not touch the `tbq*`/TurboQuant-related private branches or copy their
  contents into public branches — that code is proprietary and deliberately
  not part of the public source.

## Contributing to upstream llama.cpp

Upstream **does not accept predominantly AI-generated pull requests** and
requires disclosure of any AI assistance. Their full policy:
[upstream CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
and [upstream AGENTS.md](https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md).

If you are preparing a change here with the intent of sending it upstream,
follow *their* rules for that submission — what's acceptable in this fork is
not automatically acceptable there. `UPSTREAM_PR_NOTES.md` tracks anything
staged for upstream.

## Build/test references

- [Build documentation](docs/build.md)
- [Server development documentation](tools/server/README-dev.md)
- Verify perplexity/perf when touching inference paths (`llama-perplexity`,
  `llama-bench`); run `test-backend-ops` when touching ggml operators.
