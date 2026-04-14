
                info = val.get_info()
                if info.get("residual") is not None and info["residual"] > val.max_residual:
                    fails.append("R")
                if hasattr(val, '_vel_validator') and not val._vel_validator.is_valid():
                    fails.append("V")
                if hasattr(val, '_check_shape_descriptors') and not val._check_shape_descriptors():
                    fails.append("S")
                
                if not fails and val.is_valid():
                    label += " OK"