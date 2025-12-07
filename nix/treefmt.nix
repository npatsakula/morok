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
    # TOML
    taplo = {
      enable = true;
      settings = {
        reorder_keys = true;
        reorder_arrays = true;
        reorder_inline_tables = true;
      };
    };
  };
}
