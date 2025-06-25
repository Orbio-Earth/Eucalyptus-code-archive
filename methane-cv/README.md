# Methane CV

Methane detection using Computer Vision.

## Running tests

Tests are configured using pytest. To run tests in your compute, ensure your authentication is working properly from within the conda environment.

You may use `xargs` to inject the SSO environment variables into your shell: `env $(xargs < /etc/environment.sso`. If encountering issues, simply `export FOO=bar` one by one each environment variable.

**Run tests:**

```bash
make tests
```