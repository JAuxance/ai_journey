# NumPy Journal - Training exercises

## Goal

This list follows directly the concepts from `numpy_journal.py` to help you practice progressively.

## Work rules

- Use only `import numpy as np` (and `math` if needed).
- One exercise at a time, then verify the outputs.
- Write readable code with clear variable names.
- Check `shape`, `ndim` and `dtype` when relevant.

---

## Level 1 - Fundamentals

### Exercise 1 - Observe an array
Create `a = np.array([[2, 4, 6], [1, 3, 5]])`, then display:
- `a.shape`
- `a.ndim`
- `a.size`
- `a.dtype`

### Exercise 2 - Safe copy
Create an `original` array, make a copy with `np.copy`, modify the copy and prove that `original` does not change.

### Exercise 3 - Quick creation
Create the following arrays:
- `np.arange(15)`
- `np.linspace(0, 1, num=5)`
- `np.ones((2, 3), dtype=np.int64)`
- `np.empty(4)`

### Exercise 4 - Sort and concatenate
With `x = np.array([9, 2, 7, 2, 1])`:
- sort `x`
- concatenate `x` with `np.array([100, 200])`

### Exercise 5 - Valid reshape
Starting from `b = np.arange(1, 13)`:
- reshape to `(3, 4)`
- reshape to `(2, 6)`
- explain why `(5, 3)` does not work

### Exercise 6 - Rows, columns, axes
With `t = np.array([10, 20, 30, 40])`:
- create a row version `(1, 4)` with `np.newaxis`
- create a column version `(4, 1)` with `np.newaxis`
- redo it with `np.expand_dims`

### Exercise 7 - Indexing and slicing
With `c = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])`, display:
- the element `30`
- the first row
- the second row via slicing
- the second column

### Exercise 8 - Boolean masks
With `d = np.array([[1, 8, 3], [4, 11, 6]])`, get:
- values `>= 5`
- even values
- the position of `11` with `np.nonzero`

---

## Level 2 - Intermediate

### Exercise 9 - Vertical and horizontal stacking
Create two `2x2` matrices and compare the result of:
- `np.vstack((m1, m2))`
- `np.hstack((m1, m2))`
Then give the resulting shapes.

### Exercise 10 - Matrix splitting
Create `z = np.arange(1, 29).reshape(4, 7)` then use `np.hsplit(z, (2, 4))`.
Display each sub-array with its index.

### Exercise 11 - Per-axis operations
Create `m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])`:
- compute `m.sum(axis=0)`
- compute `m.sum(axis=1)`
- explain in one sentence the difference between `axis=0` and `axis=1`

### Exercise 12 - Transpose
Generate a random `3x2` matrix and display its transpose with `.T`.
Check that the shapes are swapped.

### Exercise 13 - Unique values
With `u = np.array([4, 4, 2, 9, 2, 1, 9, 9])`:
- get the unique values
- get the indices of first occurrence (`return_index=True`)

### Exercise 14 - Column reversal
Create a `6x4` matrix then reverse the column order with `np.flip(..., axis=1)`.

---

## Level 3 - Putting it into practice

### Exercise 15 - Pipeline 1D to 2D
Write a mini-script that:
1. creates an array from `1` to `24`
2. reshapes it to `6x4`
3. displays the 3rd row
4. displays only the even values
5. reverses the column order

### Exercise 16 - Flatten vs ravel
Show with an example that:
- `flatten()` returns a copy
- `ravel()` can share memory with the original

### Exercise 17 - Automatic array summary
Write a function `describe_array(arr)` that displays:
- `shape`, `ndim`, `size`, `dtype`
- minimum, maximum, mean

### Exercise 18 - Final mini challenge
From a random `5x5` matrix:
1. replace all values `< 30` with `0`
2. extract values `>= 30`
3. display the positions of values strictly greater than `80`

---

## Recommended submission format (clean for the repo)

1. Create a file `numpy_journal_solutions.py`.
2. One function per exercise: `exo_01()`, `exo_02()`, etc.
3. Add a `main()` that calls only the completed exercises.
4. Add a comment at the top of the file with:
   - date
   - progress (`Level 1`, `Level 2`, `Level 3`)
5. Make a commit with a clear message, for example:
   - `add numpy journal training exercises solutions (level 1)`

## Validation criteria

- All Level 1 exercises run without errors.
- At least 4 Level 2 exercises completed.
- 1 Level 3 mini challenge completed.
- Code pushed to the repo with a clean commit.
