{
  description = "habitctl";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    cargo2nix.url = "github:cargo2nix/cargo2nix/feature/add-basic-flake-support";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, cargo2nix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [
          (import rust-overlay)
          (import "${cargo2nix}/overlay")
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
      in
      with pkgs;
      {
        devShell = mkShell {
          buildInputs = [
            rust-bin.stable.latest.default
            cargo2nix.defaultPackage.${system}
          ];
        };

        defaultPackage =
          (pkgs.rustBuilder.makePackageSet' {
            rustChannel = "stable";
            packageFun = import ./Cargo.nix;
          }).workspace.habitctl {};
      });
}
