See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Quick development

The fastest way to start with development is to use Pixi. If you don't have
Pixi, you can learn to install it
[here](https://pixi.prefix.dev/latest/installation/)

After you have installed Pixi, install the `dev` environment:

```bash
pixi install -e dev
```

# Pre-commit

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pixi run check` to check even without installing
the hook.

# Testing

Use Pixi to run the unit checks:

```bash
pixi run tests
```
