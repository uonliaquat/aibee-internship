# Multi-Head Attention

## Build (Debug)

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
````

## Build (Release)

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## For Testing
### 1) Build Shared Library Only

```bash
cd build
cmake ..
make
```

### 2) Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install Python Dependencies

```bash
pip install numpy torch
```


### 4) Run Python Test

```bash
python3 test/test_attention.py
```

## Project Structure

```
project/
├── Inference Engine/
├── blis/
└── Readme.md

```
