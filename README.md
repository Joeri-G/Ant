# ANT - A Non-monetary sharing economy analysis toolkiT
---
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/Joeri-G/Ant/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/Joeri-G/Ant)
[![SWH](https://archive.softwareheritage.org/badge/swh:1:dir:43b988c3653ca1b9b7498c4e6078914b26b95f87/)](https://archive.softwareheritage.org/swh:1:dir:43b988c3653ca1b9b7498c4e6078914b26b95f87;origin=https://github.com/Joeri-G/Ant;visit=swh:1:snp:c45c352d86ddbe9d72f2db89beb258127d8ce756;anchor=swh:1:rev:9dabed9d13c36fa61552b122c376b4e9ec906201)

## Installation

Your best bet is to install everything using the [`nix`](https://nixos.org) package manager. Don't confuse the OS with the package manager.
If you go this route, you can start a dev env with all the required dependencies with
```bash
$ nix develop
```

If `nix` is not an option, everything should work with `uv`, or even plain python if you install:
 - `python3.1x`
 - `numpy`
 - `cvxpy`
 - `networkx`

No guarantees, though.

## Usage

- Generate a market
- Calculate a theoretical optimum with a convex solver
- Simulate markets

## Documentation

Important components have docstrings.

Take a look at the experiments in the [experiments](experiments) folder.

## Known problems

There are no known problems if I refuse to look for them.
