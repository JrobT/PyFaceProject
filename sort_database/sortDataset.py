#!/usr/bin/env python3

"""Python script to combine datasets.

I needed my dataset to be large enough to justify my results,
therefore I decided I would combine several sources of images together.
"""

import os
import time


# Start the script.
script_name = os.path.basename(__file__)  # The name of this script
print("\nBeginning to sort the dataset...".format(script_name))
start = time.clock()  # Start of the speed test. ``clock()'' is most accurate.

with open("sortCKplus.py") as fname:
    code = compile(fname.read(), "sortCKplus.py", 'exec')
    exec(code)

# with open("sortMMI.py") as fname:
#     code = compile(fname.read(), "sortMMI.py", 'exec')
#     exec(code)

with open("sortJAFFE.py") as fname:
    code = compile(fname.read(), "sortJAFFE.py", 'exec')
    exec(code)

with open("sortIMM.py") as fname:
    code = compile(fname.read(), "sortIMM.py", 'exec')
    exec(code)

# with open("sortGoogle.py") as fname:
#     code = compile(fname.read(), "sortGoogle.py", 'exec')
#     exec(code)

with open("sortRafD.py") as fname:
    code = compile(fname.read(), "sortRafD.py", 'exec')
    exec(code)

# End the script.
end = time.clock()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("\n***> Time elapsed: {:0>2}:{:0>2}:{:05.2f}."
      .format(int(hours), int(minutes), seconds))
