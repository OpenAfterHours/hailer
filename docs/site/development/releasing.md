# Releasing {#releasing}

```bash
uv run python -m scripts.release            # patch bump: 0.1.0 -> 0.1.1
uv run python -m scripts.release minor      # 0.1.0 -> 0.2.0
uv run python -m scripts.release major      # 0.1.0 -> 1.0.0
uv run python -m scripts.release --dry-run  # preflight checks and the plan, nothing changed
```

Run it from a clean `main` that matches `origin/main`. The script writes the new version to `pyproject.toml`,
`src/hailer/__init__.py` and `uv.lock`, then runs the test suite. If the tests fail, the version files are
restored and nothing is committed. If they pass, it commits `Release vX.Y.Z`, tags `vX.Y.Z` and pushes the
branch and tag atomically. The tag triggers `.github/workflows/release.yml`, which:

1. runs every job in `test.yml` again (the four platforms and the Docker kernel job) and builds the sdist
   and wheel;
2. makes sure the kernel image of this Hailer's kernel contract,
   `ghcr.io/openafterhours/hailer-kernel:marimo<version>-<fingerprint>`, is in the registry: when the tag is
   already published (the release changed nothing in the image) nothing is built; otherwise
   `scripts/build_kernel_image.py --push --if-missing` builds it for `linux/amd64` and `linux/arm64` and
   pushes it. That step is the one gate: a tag published without both platforms, only some tags published,
   or any registry error other than "not found" fails the job without overwriting anything;
3. only when all of that succeeds, publishes to PyPI through the `pypi` environment (trusted publishing,
   no token to store) and creates the GitHub release with the files attached. The image goes first so no
   released Hailer points at a missing image.

**Changing the kernel image.** The image is defined by `src/hailer/docker/Dockerfile`, the modules in
`hailer.kernel_image.IMAGE_MODULES` (`errors.py`, `periods.py`, `_forward.py`) and the exact versions in
`IMAGE_PACKAGES`. The tag is computed from them (`marimo<version>-<first 12 hex of their sha256>`, line
endings ignored), so any edit to one of them, even a comment, gives a new tag, and the next release builds
and publishes it automatically; there is nothing to bump. Upgrading marimo means changing its pin in
`pyproject.toml` and `MARIMO_VERSION` together (a test checks they agree).

`--no-push` stops after the local commit and tag, `--version X.Y.Z` releases an exact version (pre-releases
such as `1.2.0rc1` are accepted), and arguments after `--` are passed to pytest.

**One-time step for the kernel image.** GHCR creates the `hailer-kernel` package as private on the first
push. After the first release, make it public in the organisation's package settings
(github.com/orgs/OpenAfterHours/packages, `hailer-kernel`, *Package settings*, *Change visibility*);
otherwise `uvx hailer kernel pull` and the first docker start fail for everyone outside the organisation
(they see `The kernel image marimo<version>-<fingerprint> is not published (or not visible to you)`). If a
`hailer-kernel` package was ever pushed by hand, also give this repository the *Write* role under *Manage
Actions access* on the same page, or the workflow's push is refused.

**A new kernel contract has no published image until it is released.** Build one for the checkout into
your local Docker with `uv run python -m scripts.build_kernel_image --load`. It uses the build context
`uvx hailer kernel build` and the release use (the packaged Dockerfile and the image modules of this checkout); `--dry-run` prints
the `docker buildx build` command, `--tag` renames the image, and `--help` lists the rest.

Repository rulesets restrict this: `main` cannot be force-pushed or deleted and changes to it must come
through a pull request with the test checks green, and `v*` tags can only be created by repository admins,
who also bypass the pull-request rule so the release script can push directly. The `pypi` environment only
deploys from `v*` tags, and `.github/workflows/members-only.yml` closes pull requests opened from forks by
people outside the OpenAfterHours organization. Open your own pull requests from a branch in this repository;
those are always kept.
