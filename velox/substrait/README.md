# Velox substrait upgrade

Velox includes a copy of the Substrait proto files from `https://github.com/substrait-io/substrait/tree/main/proto/substrait` to generate substrait cpp files. So if you want to upgrade substrait to a certain version, follow the steps below:

- Copy substrait proto files for a certain version in `https://github.com/substrait-io/substrait/tree/main/proto/substrait` to velox substrait proto dir: `https://github.com/facebookincubator/velox/tree/main/velox/substrait/proto/substrait`
- Add `option cc_enable_arenas = true;` to the top of option definition area for every proto file

After the new files are copied, ensure that the new code compiles and tests pass, then submit a PR.
