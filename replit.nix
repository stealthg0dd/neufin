{pkgs}: {
  deps = [
    pkgs.zip
    pkgs.rustc
    pkgs.libiconv
    pkgs.cargo
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.glibcLocales
  ];
}
