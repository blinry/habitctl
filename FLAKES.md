## flakes

https://nixos.wiki/wiki/Flakes

## dev shell

```
nix develop
```

## build project

```
nix build
```

run the executable:

```
./result/bin/habitctl
```

## keeping `Cargo.nix` up to date

this should be run if `Cargo.toml` / `Cargo.lock` are changed. `Cargo.nix` is used for `nix build`.

```
cargo2nix -f
```
