# Arrow ADBC Driver Manager

Vendored copy of the Apache Arrow ADBC driver manager, taken from the
[apache/arrow-adbc](https://github.com/apache/arrow-adbc) repository at
release tag `apache-arrow-adbc-14`:

- `arrow-adbc/adbc.h` — the ADBC C API (`c/include/arrow-adbc/adbc.h`).
- `arrow-adbc/adbc_driver_manager.h` — driver manager API
  (`c/include/arrow-adbc/adbc_driver_manager.h`).
- `adbc_driver_manager.cc` — driver manager implementation
  (`c/driver_manager/adbc_driver_manager.cc`).

The driver manager loads ADBC driver shared libraries at runtime (via
`dlopen`) and exposes the uniform ADBC C API. Release 14 is the newest tag
whose driver manager has no third-party dependencies; later releases add
driver-manifest support that requires toml++.

Licensed under the Apache License 2.0 (see `LICENSE.txt`).
