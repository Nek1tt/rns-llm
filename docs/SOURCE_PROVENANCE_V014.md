# Source provenance

v0.14.2 объединяет предоставленные пользователем реализации:

- full-RNS v0.7 (`v07_extension.cu`, `v07_backend.py`, `FastRNSLinearV07`);
- hybrid/prefill v0.11.3 (`v011_prefill_extension.cu`, `PrefillLayerV011`);
- full-RNS precision/LUT study v0.13 (`v013_architecture_extension.cu`).

Изменения v0.14.2:

- actual-size compact LUT allocation in hybrid;
- shared LUT cache across stream-specific runners;
- q32 hybrid support through 128-bit Garner and up to 10 channels;
- mode-specific cuBLASLt workspaces;
- native INT8 separated from hybrid;
- unified matrix/Attention/PPL comparison;
- version-compatible NSYS capture-range handling; complete NSYS SQLite/schema/queries/JSON pipeline; NCU article-essential raw/details JSON with optional exhaustive mode.
