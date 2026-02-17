"""Remove orphaned old code block from app.py (lines ~2110-2162)."""
import pathlib

p = pathlib.Path(r"c:\WORK\Zenith_App\app.py")
raw = p.read_text(encoding="utf-8-sig")
lines = raw.splitlines(keepends=True)

# Find the orphaned block boundaries:
# Start: first line after `st.markdown` at module level that has `"<div class='m-head'>`
# End: the duplicate `st.markdown` line right before `def _add_hline`
start_del = None
end_del = None

for i, line in enumerate(lines):
    # We already passed the new function body (ends with st.markdown at line ~2105)
    # The orphaned block starts around line 2110 with indented string literals at module level
    if i >= 2107 and start_del is None:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('def '):
            start_del = i
    if start_del is not None and end_del is None:
        if line.strip().startswith('def _add_hline'):
            end_del = i
            break

if start_del is not None and end_del is not None:
    print(f"Removing lines {start_del+1} to {end_del} (0-indexed: {start_del}:{end_del})")
    new_lines = lines[:start_del] + ["\n", "\n"] + lines[end_del:]
    p.write_text("".join(new_lines), encoding="utf-8")
    print(f"Done: {len(lines)} -> {len(new_lines)} lines")
else:
    print(f"Could not find block: start_del={start_del}, end_del={end_del}")
    print("Listing lines around 2108-2170:")
    for i in range(2105, min(2170, len(lines))):
        print(f"  {i+1}: {lines[i].rstrip()[:80]}")
