# NumPy knowledge test

## Instructions

- Suggested duration: 45 to 60 minutes
- Answer directly under each question or in a separate Python file
- Use only the concepts present in `numpy_journal.py`
- Total scoring: 20 points

---

## Part 1 - NumPy array basics (4 points)

### Exercise 1 - Vocabulary and observation (2 points)

Consider the following array:

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
```

1. Give the value of `a.shape`
2. Give the value of `a.ndim`
3. Give the value of `a.size`
4. Explain in one sentence the difference between `shape` and `size`

### Exercise 2 - Copy (2 points)

The following code is executed:

```python
original = np.array([[1, 2], [3, 4]])
copy = np.copy(original)
copy[0, 0] = 99
```

1. What is the value of `copy[0, 0]`?
2. What is the value of `original[0, 0]`?
3. Why is `np.copy()` useful here?

---

## Part 2 - Array creation and transformation (6 points)

### Exercise 3 - Array creation (2 points)

Write a NumPy one-liner to create:

1. An array containing the integers from `0` to `11`
2. An array of 6 evenly spaced values between `10` and `100`
3. An array of three `1`s with type `int64`

### Exercise 4 - Reshape and axes (2 points)

Consider:

```python
b = np.array([[1, 2, 3], [4, 5, 6]])
```

1. Reshape `b` into an array of shape `(3, 2)`
2. Explain why this transformation is possible
3. From `t = np.array([1, 2, 3, 4])`, create:
   - a row version of shape `(1, 4)`
   - a column version of shape `(4, 1)`

### Exercise 5 - Flatten and ravel (2 points)

Answer the following questions:

1. What behavior difference is observed between `flatten()` and `ravel()`?
2. Which one returns a copy?
3. Which one can modify the original array if you modify the result?

---

## Part 3 - Indexing, slicing and masks (4 points)

### Exercise 6 - Reading values (2 points)

Consider:

```python
c = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
```

Give the result of:

1. `c[1, 2]`
2. `c[0]`
3. `c[1:2]`
4. `c[:, 1]`

### Exercise 7 - Boolean masks (2 points)

Consider:

```python
d = np.array([[1, 8, 3], [4, 11, 6]])
```

1. Write an expression that selects values greater than or equal to `5`
2. Write an expression that selects even values
3. What is the purpose of `np.nonzero(d == 11)`?

---

## Part 4 - Assembly, splitting and operations (6 points)

### Exercise 8 - Stacking and splitting (2 points)

Consider:

```python
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
z = np.arange(1, 29).reshape(4, 7)
```

1. What is the difference between `np.vstack((x, y))` and `np.hstack((x, y))`?
2. What does `np.hsplit(z, (2, 4))` do?
3. How many sub-arrays are produced by this operation?

### Exercise 9 - Useful computations and operations (2 points)

Consider:

```python
e = np.arange(1, 11)
f = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
```

1. What does the expression `f // e` do?
2. What does `m.sum(axis=0)` return?
3. What does `m.sum(axis=1)` return?
4. What is `.T` used for on a matrix?

### Exercise 10 - Uniques, reversal and mini-code (2 points)

1. What is `np.unique(tab, return_index=True)` used for?
2. What is `np.flip(tab, axis=1)` used for?
3. Write a short NumPy snippet that:
   - creates an array of 24 integers
   - reshapes it to a 6 rows by 4 columns matrix
   - reverses the column order

---

## Bonus - Practical exercise (2 points)

Write a NumPy script that performs all the following steps:

1. Create a 1D array containing the values from `1` to `12`
2. Reshape it to a `3 x 4` matrix
3. Display the second row
4. Display only the even values
5. Stack this matrix horizontally with itself
