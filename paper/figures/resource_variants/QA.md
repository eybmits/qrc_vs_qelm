# Selection-board QA

All three exported selection strips and the nine-panel overview were inspected.
No clipped labels, legends or inset titles were observed. The overview is an
index; use the individual images or row strips for detailed selection.

- The 912 original root/model rows remain unchanged and hash-verified.
- Ten variant tests check coverage, quantiles, axis scales and exact displayed
  medians against the source table.
- The main and conference PDFs, current Figure 2 PDF and source CSV retain
  their pre-variant hashes, recorded in manifest.json.
- The full regression run passed 759 tests; its sole failure was the release
  inventory missing the two new presentation code/test files. The inventory
  was regenerated without changing any prior artifact hash, then the inventory
  check and ten variant tests were rerun.
- Ruff and whitespace checks pass. No experiments were executed in this task.

These are alternative designs, not adopted manuscript replacements. Insets
and faceted versions should receive a final typography check at the eventual
paper footprint after the author chooses a combination.
