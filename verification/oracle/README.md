# Captured oracle artefacts live here.
#
# Subdirectories are versioned by the cuQuantum SDK they were captured against:
#
#   oracle/v1.0.0-cuquantum-cu13/
#       custatevec/
#           apply_matrix/
#               <param-id>.npz
#               <param-id>.json
#       cutensornet/
#       ...
#
# Use Git LFS for *.npz so the oracle does not bloat the main repo.
# Suggested .gitattributes lines (place at repo root or in this subtree):
#
#   verification/oracle/**/*.npz filter=lfs diff=lfs merge=lfs -text
#
# Captures are produced by ``verification/capture/capture_<lib>.py`` on a
# host with the cuQuantum SDK installed. They are consumed by
# ``verification/replay/replay_<lib>.py`` on a host with rocQuantum installed.
