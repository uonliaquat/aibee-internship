````markdown
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

### Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Python Dependencies

```bash
pip install numpy torch
```


### Run Python Test (Seprate)

```bash
python3 test/test_attention.py
python3 test/test_linear.py
python3 test/test_layernorm.py
python3 test/test_add.py

python3 test/test_all.py
```

### Run Python Test (All)

```bash
python3 test/test_all.py
```

## Project Structure

```
project/
├── Inference Engine/
├── blis/
└── Readme.md

```

```
```
