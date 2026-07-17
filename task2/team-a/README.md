# Task 2

## Team-A

## Kernels Implemented

#### add
- It takes two pointers of floating-point numbers and adds them together element by element. 
- returns the output in a third output pointer, it is not an in place addition

#### layernorm
- formula = `((x - mean) / sqrt(var + eps)) * gamma + beta`
- We find the mean, find the variance, subtract the mean from the input, divide by the standard deviation, and finally scale and shift the result using learned weights.
- __restrict keyword on the input and output pointers, telling it the arrays dont overlap so it can safely and aggressively optimize the loops.
- When you add up floating-point numbers to calculate the mean and variance accuracy can suffer so we typecasted into double to keep maths stable
- compared to mul division is alot slower, so calculate inverse and instead of div we multiplited with inv instead 
- fmaf is fused multiplication addition, it does `a*b+c` in one clock. using it for mul and adition speeds up calculation significantly

## How To Run

to build 
```
make
```
to run tests
```
make test
```
to clean everyting
```
make clean 
```