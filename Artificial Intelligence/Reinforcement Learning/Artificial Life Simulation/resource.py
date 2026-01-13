
# resource.py

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO HPE: Handle possible error.
#TODO AC: Add comment.
#TODO AIC: Add input control.

#! PW: Possibly wrong.

class ResourceManager: #TODO AD
    
    resource_colors: dict[str, tuple[int, int, int]] = {
        "food": (150, 75, 0), # Brown.
        "corpse": (255, 165, 0), # Orange.
        "waste": (0, 100, 0) # Dark Green.
    }
    