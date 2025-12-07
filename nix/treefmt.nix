{ ... }:
{
  projectRootFile = "flake.nix";
  programs = {
    mdformat = {
      enable = true;
      settings.number = true;
    };
  };
}
