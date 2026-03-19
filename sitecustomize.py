# sitecustomize.py
# Copied to .venv/Lib/site-packages/ so it loads automatically for this venv.
# Patches Phi-3.5 rope_scaling so 'longrope' type doesn't crash on load.

def _patch_phi3_rope():
    try:
        from transformers.models.phi3.configuration_phi3 import Phi3Config

        if getattr(Phi3Config._rope_scaling_validation, "_patched", False):
            return

        _original = Phi3Config._rope_scaling_validation

        def _patched(self):
            rs = getattr(self, "rope_scaling", None)
            if isinstance(rs, dict) and rs.get("type") not in ("su", "yarn"):
                # Try yarn first, fall back to su
                for rope_type in ("yarn", "su"):
                    self.rope_scaling = {**rs, "type": rope_type}
                    try:
                        return _original(self)
                    except ValueError:
                        continue
                # Both failed — restore original and let it raise naturally
                self.rope_scaling = rs
            return _original(self)

        _patched._patched = True
        Phi3Config._rope_scaling_validation = _patched

    except Exception:
        pass


_patch_phi3_rope()
