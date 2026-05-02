# Testing Guide

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run specific kernel tests
```bash
# Dense kernel tests only
pytest tests/test_sigmoid_dense.py -v

# Padded kernel tests only
pytest tests/test_sigmoid_padded.py -v
```

### Run specific test cases
```bash
pytest tests/test_sigmoid_dense.py::test_dense_op -k "float16 and fwd"
pytest tests/test_sigmoid_padded.py::test_op -k "dtype0-fwd-64-128"
```
