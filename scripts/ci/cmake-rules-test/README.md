# CMake Linter Rule Tests

This directory contains test configuration files to verify that CMake linting rules work correctly.

## Test Files

### deprecated-link-directories-test.yml
A test configuration file that validates the `deprecated-link-directories` rule which detects usage of the deprecated `link_directories()` command.

The test file contains:
- **valid** cases: CMake code that should NOT trigger the rule (proper usage patterns)
- **invalid** cases: CMake code that SHOULD trigger the rule (deprecated patterns)

## Running Tests

To run all rule tests:
```bash
ast-grep test
```
## Writing Test Cases

Test configuration files follow this structure:

```yaml
id: rule-name-to-test
valid:
  # Code examples that should NOT trigger the rule
  - |
    target_link_libraries(mylib PRIVATE /usr/local/lib/libfoo.so)
  - |
    find_library(FOO_LIB foo PATHS /usr/local/lib)
    target_link_libraries(mylib PRIVATE ${FOO_LIB})
invalid:
  # Code examples that SHOULD trigger the rule  
  - link_directories(/usr/local/lib)
  - |
    link_directories(
        /opt/lib
        /usr/lib64
    )
```
