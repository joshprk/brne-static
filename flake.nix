{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs: inputs.flake-parts.lib.mkFlake {inherit inputs;} {
    perSystem = {pkgs, ...}: let
      python = pkgs.python3;

      project = inputs.pyproject-nix.lib.project.loadPyproject {
        projectRoot = ./.;
      };

      attrs = project.renderers.withPackages {
        inherit python;
      };

      pythonEnv = python.withPackages attrs;
    in {
      devShells.default = pkgs.mkShell {
        packages = [
          pkgs.typst
          pythonEnv
        ];
      };
    };

    systems = ["x86_64-linux"];
  };
}
