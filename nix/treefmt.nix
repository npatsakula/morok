{ ... }:
{
  projectRootFile = "flake.nix";
  programs = {
    mdformat = {
      enable = true;
      settings.number = true;
    };
    nixfmt.enable = true;
    yamlfmt = {
      enable = true;
      settings.formatter.retain_line_breaks = true;
    };
  };
}
