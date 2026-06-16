# NumPy - Topics to review

## High priority

- Boolean mask syntax: `tab[condition]`
- Difference between `axis=0` and `axis=1`
- Answer exactly the question when a single value is requested, such as `a[0, 0]`

## Syntax to secure

- `np.linspace(...)` not `np.linespace(...)`
- `np.ones(3, dtype=np.int64)` with the `np.` prefix
- Verify the number of elements before a `reshape`

## Concepts to revisit

- `np.nonzero(...)` to find a position
- `np.unique(tab, return_index=True)` to get unique values and their first position
- `np.flip(tab, axis=1)` to reverse the column order
- `.T` to transpose a matrix
- `np.hsplit(...)` to split along columns
